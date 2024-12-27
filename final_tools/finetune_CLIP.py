import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open_clip
from dataset_class import ImageTitleDataset

model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
tokenizer = open_clip.get_tokenizer('ViT-B-16')

# Gerät auswählen
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# Dataset erstellen
#root_dir = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\building_surround_pictures"
root_dir = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\building_big_surround_pictures"
dataset = ImageTitleDataset(root_dir, transform=preprocess, device=device)
train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

# Optimizer und Loss-Funktionen
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Dataset splitten in Training und Validation
train_size = 0.8  # Anteil der Daten für das Training
val_size = 0.2    # Anteil der Daten für die Validierung
train_len = int(len(dataset) * train_size)
val_len = len(dataset) - train_len
print(dataset.__len__())
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

# DataLoader für Trainings- und Validierungsdaten
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Funktion zur Berechnung der Genauigkeit
def calculate_accuracy(logits, labels):
    """
    Berechnet die Genauigkeit basierend auf den Logits und den Ground-Truth-Labels.
    :param logits: Vorhersagen des Modells
    :param labels: Ground-Truth-Labels
    :return: Genauigkeit (Accuracy)
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

# Training und Validierung
num_epochs = 2
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_accuracy = 0
    pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}/{num_epochs}")
    for batch in pbar:
        optimizer.zero_grad()
        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Loss berechnen
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        # Backward pass
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_accuracy += calculate_accuracy(logits_per_image, ground_truth)

        pbar.set_postfix({'Loss': total_loss.item(), 'Accuracy': train_accuracy / len(pbar)})

    train_loss /= len(train_dataloader)
    train_accuracy /= len(train_dataloader)

    # Validierung
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        val_pbar = tqdm(val_dataloader, total=len(val_dataloader), desc="Validation")
        for batch in val_pbar:
            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Loss berechnen
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            val_loss += total_loss.item()
            val_accuracy += calculate_accuracy(logits_per_image, ground_truth)

            val_pbar.set_postfix({'Loss': total_loss.item(), 'Accuracy': val_accuracy / len(val_pbar)})

    val_loss /= len(val_dataloader)
    val_accuracy /= len(val_dataloader)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Modell speichern
torch.save(model.state_dict(), "openclip_finetuned_with_validation.pth")