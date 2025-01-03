import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPProcessor, CLIPModel
import json
from torchvision.transforms import Resize

# Set random seed for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define augmentation and preprocessing
preprocess = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275)),
])

# Define label vocabulary
building_vocab = {}

def build_vocab(labels):
    global building_vocab
    unique_labels = set()
    for label in labels:
        unique_labels.update(label.split(" and "))
    building_vocab = {label: idx for idx, label in enumerate(sorted(unique_labels))}

def encode_labels(labels):
    """
    Convert string labels into binary vectors for multi-label classification.
    """
    global building_vocab
    encoded = torch.zeros(len(building_vocab))
    for label in labels.split(" and "):
        if label in building_vocab:
            encoded[building_vocab[label]] = 1
    return encoded

# Define custom Dataset class
class WholeSceneJsonDataset(Dataset):
    def __init__(self, root_dir, preprocess, limit_data=False, limit_count=1000):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.image_label_pairs = []
        self.resize = Resize((224, 224))

        # Load all JSON files and extract labels
        for root, dirs, files in os.walk(root_dir):
            for file in tqdm(files, desc="Loading JSON files"):
                if file.endswith(".frame_data.json"):
                    json_path = os.path.join(root, file)

                    with open(json_path, "r") as f:
                        data = json.load(f)

                    image_filename = data["captures"][0]["filename"]
                    image_path = os.path.join(root, image_filename)

                    try:
                        instances = data["captures"][0]["annotations"][0]["instances"]
                        labels = " and ".join([instance["labelName"] for instance in instances])
                    except (KeyError, IndexError) as e:
                        print(f"Error processing JSON: {json_path} - {e}")
                        labels = "Unknown"

                    if os.path.exists(image_path):
                        self.image_label_pairs.append((image_path, labels))

        if limit_data:
            random.shuffle(self.image_label_pairs)
            self.image_label_pairs = self.image_label_pairs[:limit_count]

        # Build the vocabulary for all labels
        all_labels = [label for _, label in self.image_label_pairs]
        build_vocab(all_labels)

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_path, label = self.image_label_pairs[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        image = self.resize(image)

        encoded_label = encode_labels(label)
        return image, encoded_label

# Set paths
image_path = r"C:\Users\Lukas\AppData\LocalLow\DefaultCompany\Fuwa_HDRP\usefull_data\solo_2"

# Create dataset and DataLoader
dataset = WholeSceneJsonDataset(image_path, preprocess=preprocess)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Prepare optimizer with LLRD
def get_layerwise_lr_decay(model, base_lr=1e-3, decay=0.8):
    param_groups = []
    layer_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            lr = base_lr * (decay ** layer_idx)
            param_groups.append({"params": param, "lr": lr})
            if "transformer" in name or "encoder" in name:
                layer_idx += 1
    return param_groups

learning_rate = 1e-3
decay_factor = 0.8
param_groups = get_layerwise_lr_decay(model, base_lr=learning_rate, decay=decay_factor)
optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.05)

scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Define EMA class
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.detach())

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(param.detach())

# Initialize EMA
ema = EMA(model)

# Define loss function for multi-label classification
loss_fn = nn.BCEWithLogitsLoss()
epoch_losses = []

# Training loop
num_epochs = 100
best_loss = float('inf')
epoch_losses = []

for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    batch_losses = []

    for batch in pbar:
        optimizer.zero_grad()

        images, texts = batch
        images, texts = images.to(device), texts.to(device)

        # Text- und Bild-Features extrahieren
        image_features = model.get_image_features(images)  # [batch_size, feature_dim]
        text_features = model.get_text_features(texts)     # [batch_size, feature_dim]

        # Normalisieren der Features für Vergleichbarkeit (optional)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine Similarity zwischen Bild- und Text-Features
        similarity = torch.matmul(image_features, text_features.T)  # [batch_size, batch_size]

        # Zielwerte erstellen: Diagonale = 1 (Perfekte Übereinstimmung)
        targets = torch.eye(similarity.size(0)).to(device)  # [batch_size, batch_size]

        # Verlust berechnen (z. B. BCEWithLogitsLoss oder MSELoss)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(similarity, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # EMA-Update
        ema.update()

        batch_losses.append(loss.item())
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

    scheduler.step()

    # Durchschnittlicher Verlust pro Epoche
    epoch_avg_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(epoch_avg_loss)
    print(f"Epoch {epoch} Average Loss: {epoch_avg_loss:.4f}")

    # Modell speichern, wenn der Verlust sinkt
    if epoch_avg_loss < best_loss:
        best_loss = epoch_avg_loss
        ema.apply_shadow()
        torch.save(model.state_dict(), "best_clip_model.pth")
        ema.restore()

# Verlust über die Epochen plotten
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), epoch_losses, marker='o', label='Average Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Average Loss per Epoch")
plt.legend()
plt.grid()
plt.show()