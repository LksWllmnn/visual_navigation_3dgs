import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import clip
from transformers import CLIPProcessor, CLIPModel
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

class WholeSceneJsonDataset(Dataset):
    def __init__(self, root_dir, preprocess, limit_data=False, limit_count=1000):
        """
        Args:
            root_dir (str): Root directory containing image and JSON files.
            preprocess (function): Preprocessing function for the images.
            limit_data (bool): Whether to limit the dataset size.
            limit_count (int): Number of samples to include if limiting data.
        """
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.image_label_pairs = []

        # Iterate over all directories and collect image-label pairs
        for root, dirs, files in os.walk(root_dir):
            for file in tqdm(files, desc="Loading JSON files"):
                if file.endswith(".frame_data.json"):
                    json_path = os.path.join(root, file)

                    # Read the JSON file
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    # Extract the corresponding image path
                    image_filename = data["captures"][0]["filename"]
                    image_path = os.path.join(root, image_filename)

                    # Extract labels from the JSON's instances
                    try:
                        instances = data["captures"][0]["annotations"][0]["instances"]
                        labels = " and ".join([instance["labelName"] for instance in instances])
                    except (KeyError, IndexError) as e:
                        # Handle missing or malformed JSON data
                        print(f"Error processing JSON: {json_path} - {e}")
                        labels = "Unknown"

                    # Add the image path and corresponding label
                    self.image_label_pairs.append((image_path, labels))

        # Shuffle and limit data if specified
        if limit_data:
            random.shuffle(self.image_label_pairs)
            self.image_label_pairs = self.image_label_pairs[:limit_count]

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        # Get image path and label
        img_path, label = self.image_label_pairs[idx]

        # Open and preprocess the image
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)

        # Tokenize the label
        tokenized_label = clip.tokenize([label])[0]

        return image, tokenized_label
    
# Set the path to your image directory
image_path = r"F:\Studium\Master\Thesis\data\finetune\fuwa_scene_semSeg"
list_image_path = []

# Add all image paths to list_image_path
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith(".png"):  # Ensure that only .png files are included
            list_image_path.append(os.path.join(root, file))

# Create the dataset and DataLoader
dataset = WholeSceneJsonDataset(image_path, preprocess=preprocess)
train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()

# Prepare the optimizer
#learningRate=0.1
learningRate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)  # Use a small learning rate for fine-tuning

# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
epoch_losses = []

# Train the model
num_epochs = 40
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    batch_losses = []  # Store batch losses for the current epoch
    for batch in pbar:
        optimizer.zero_grad()

        images, texts = batch 
        
        images = images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        # Store the loss for this batch
        batch_losses.append(total_loss.item())

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else: 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

    # Calculate and store the average loss for the epoch
    epoch_avg_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(epoch_avg_loss)
    print(f"Epoch {epoch} Average Loss: {epoch_avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "clip_tall_20b_1e6lr_40e_wholeScene.pth")

# Plot the average loss per epoch
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), epoch_losses, marker='o', label='Average Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Average Loss per Epoch")
plt.legend()
plt.grid()
plt.show()