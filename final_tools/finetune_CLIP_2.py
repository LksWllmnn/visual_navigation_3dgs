import json
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import clip
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from dataset_class import ImageTitleDataset

#image_path = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\building_surround_pictures"
image_path = r"D:\Thesis\data\finetune_data\scene_building_pictures"


# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

    
# def plot_images_with_labels(dataset, num_images=20):
#     """
#     Plotte eine bestimmte Anzahl von Bildern mit ihren Labels aus einem Dataset.

#     :param dataset: Das Dataset, aus dem die Bilder und Labels geladen werden
#     :param num_images: Anzahl der zu plottenden Bilder
#     """
#     num_images = min(num_images, len(dataset))  # Sicherstellen, dass wir nicht mehr als das Dataset haben
#     fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 4x5 Grid für 20 Bilder
#     axes = axes.flatten()

#     for i in range(num_images):
#         image, label = dataset[i]
#         image = image.permute(1, 2, 0).numpy()  # Ändere die Dimensionen für matplotlib (HWC-Format)

#         # Zeige das Bild und den zugehörigen Text
#         axes[i].imshow(image)
#         axes[i].set_title(f"Label: {label}")
#         print(label)
#         axes[i].axis('off')  # Deaktiviere Achsenbeschriftung

#     # Blende überschüssige Subplots aus, falls weniger als 20 Bilder angezeigt werden
#     for ax in axes[num_images:]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

# def count_images_in_folder(folder_path, image_extensions=None):
#     """
#     Zählt die Anzahl der Bilder in einem Ordner.
    
#     :param folder_path: Pfad zum Ordner
#     :param image_extensions: Liste der erlaubten Bildformate (z. B. ['.jpg', '.png'])
#     :return: Anzahl der Bilder im Ordner
#     """
#     if image_extensions is None:
#         image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

#     # Konvertiere Erweiterungen in Kleinbuchstaben
#     image_extensions = [ext.lower() for ext in image_extensions]

#     # Zähle die Dateien, die eine der Erweiterungen haben
#     image_count = sum(
#         1 for file in os.listdir(folder_path)
#         if os.path.isfile(os.path.join(folder_path, file)) and os.path.splitext(file)[1].lower() in image_extensions
#     )
#     return image_count

# use your own data
# list_image_path = []
# list_txt = []
# buildings = ['A-Building', 'B-Building', 'C-Building', 'E-Building', 'F-Building',
#                  'G-Building', 'H-Building', 'I-Building', 'L-Building', 'M-Building',
#                  'N-Building', 'O-Building', 'R-Building', 'Z-Building']
# for idx in range(count_images_in_folder("F:\\Studium\\Master\\Thesis\\Unity\\Furtwangen\\Recordings_Single\\")):
#     # Gebäudeindex berechnen
#     building_idx = idx // 12  # Wechselt nach je 12 Bildern das Gebäude
#     number_idx = idx % 12  # Zahlen von 0000 bis 0011

#     # Sorge dafür, dass der index nicht über die Anzahl der Gebäude hinaus geht
#     if building_idx >= len(buildings):
#         break

#     # Generiere den Bildnamen basierend auf dem Gebäude und der laufenden Nummer
#     building = buildings[building_idx]
#     img_path = f"{image_path}Fuwa_{building}_{number_idx:04d}.png"  # Format 0000 bis 0011
#     list_image_path.append(img_path)

#     # Füge die ersten 120 Zeichen der Beschreibung hinzu
#     list_txt.append(building)


dataset = ImageTitleDataset(root_dir=image_path)

train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True) #Define your own dataloader
# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch 
        
        images= images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
    save_directory = "./chkpts/"

    # Speichere das trainierte Modell
    torch.save(model.state_dict(), f"{save_directory}scene_pictures_clip.pth")