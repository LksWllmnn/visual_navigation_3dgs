# based on https://github.com/shashnkvats/Indofashionclip/blob/main/indofashion_clip.py 14.01.2025

from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel
from dataset_class import ImageTitleDataset
import matplotlib.pyplot as plt

def calculate_accuracy(logits, labels):
    """
    Calculate accuracy based on model logits and ground-truth labels.
    :param logits: Predictions from the model
    :param labels: Ground-truth labels
    :return: Accuracy
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def train_and_evaluate(model, train_loader, val_loader, device, num_epochs=30, lr=1e-5, save_path="best_model.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_accuracy = 0.0
    best_val_loss = float('inf')  # Start with an infinitely large loss

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, texts in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, texts = images.to(device), texts.to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = (loss_fn(logits_per_image, ground_truth) + loss_fn(logits_per_text, ground_truth)) / 2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc="Validation"):
                images, texts = images.to(device), texts.to(device)
                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                loss = (loss_fn(logits_per_image, ground_truth) + loss_fn(logits_per_text, ground_truth)) / 2
                val_loss += loss.item()
                val_accuracy += calculate_accuracy(logits_per_image, ground_truth)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy / len(val_loader))

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

        # Check if the model should be saved
        if (val_accuracies[-1] > best_val_accuracy) or \
           (val_accuracies[-1] == best_val_accuracy and val_losses[-1] < best_val_loss):
            best_val_accuracy = val_accuracies[-1]
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with Validation Accuracy: {best_val_accuracy:.4f}, Validation Loss: {best_val_loss:.4f}")

    return train_losses, val_losses, val_accuracies

def save_plots_and_logs(train_losses, val_losses, val_accuracies, log_path="training_log.txt", plot_dir="plots"):
    """
    Speichert Trainings- und Validierungsplots und speichert Logs für Loss und Accuracy.
    :param train_losses: Liste der Trainingsverluste pro Epoche
    :param val_losses: Liste der Validierungsverluste pro Epoche
    :param val_accuracies: Liste der Validierungsgenauigkeiten pro Epoche
    :param log_path: Pfad zur Log-Datei
    :param plot_dir: Verzeichnis zum Speichern der Plots
    """
    # Erstelle Verzeichnis für Plots, falls nicht vorhanden
    os.makedirs(plot_dir, exist_ok=True)

    # Loss-Plot speichern
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    loss_plot_path = os.path.join(plot_dir, "loss_per_epoch.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

    # Accuracy-Plot speichern
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    accuracy_plot_path = os.path.join(plot_dir, "accuracy_per_epoch.png")
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(f"Accuracy plot saved to {accuracy_plot_path}")

    # Log-Datei speichern
    with open(log_path, "w") as log_file:
        log_file.write("Epoch\tTrain Loss\tValidation Loss\tValidation Accuracy\n")
        for epoch in range(len(train_losses)):
            log_file.write(f"{epoch+1}\t{train_losses[epoch]:.4f}\t{val_losses[epoch]:.4f}\t{val_accuracies[epoch]:.4f}\n")
    print(f"Training log saved to {log_path}")

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

#image_path = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\building_surround_pictures"
#image_path = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\scene_building_pictures"
image_path = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\building_big_surround_pictures"
#image_path = r"D:\Thesis\data\finetune_data\building_surround_pictures"
dataset = ImageTitleDataset(root_dir=image_path, filter = True)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

train_losses, val_losses, val_accuracies = train_and_evaluate(model, train_loader, val_loader, device="cuda:0", num_epochs=30, lr=1e-6)

# Logs und Plots speichern
save_plots_and_logs(train_losses, val_losses, val_accuracies, log_path="training_log.txt", plot_dir="plots")