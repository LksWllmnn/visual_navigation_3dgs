import json
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import random

import clip
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

json_path = 'path to train_data.json'
#image_path = "C:/Users/Lukas/AppData/LocalLow/DefaultCompany/Fuwa_HDRP/singleBuildings_ResnetSam"
image_path = "C:\\Users\\Lukas\\AppData\\LocalLow\\DefaultCompany\\Fuwa_HDRP\\filteredBuildings"
#image_path = "F:\\Studium\\Master\\Thesis\\Unity\\Furtwangen\\Recordings_Single\\"


# with open(json_path, 'r') as f:
#     input_data = []
#     for line in f:
#         obj = json.loads(line)
#         input_data.append(obj)


# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

class image_title_dataset():
    def __init__(self, list_image_path, list_txt):
        # Initialisiere die Bildpfade und die zugehörigen Texte
        self.image_path = list_image_path
        # Tokenisiere den Text mit dem CLIP-Tokenizer
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Bildvorverarbeitung mit CLIP's Preprocessing-Funktion
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

# class BuildingDataset():
#     # Define a custom dataset
#     def __init__(self, root_dir, buildings, transform=None, black_threshold=10, limit=None):
#         """
#         :param root_dir: Wurzelverzeichnis der Daten
#         :param buildings: Liste der zu ladenden Gebäude
#         :param transform: Transformationen für die Bilder
#         :param black_threshold: Schwelle für das Filtern schwarzer Bilder (Helligkeit)
#         :param limit: Maximale Anzahl von Daten (zufällig ausgewählt)
#         """
#         self.image_paths = []
#         self.texts = []
#         self.transform = transform
#         self.buildings = buildings
#         self.black_threshold = black_threshold

#         print("Scanning directories and filtering images...")
#         all_image_paths = []
#         all_texts = []

#         #Fortschrittsbalken für Gebäude
#         # with tqdm(total=len(buildings), desc="Processing buildings", unit="building") as building_bar:
#         #     for building in buildings:
#         #         building_path = os.path.join(root_dir, building)
#         #         if os.path.isdir(building_path):
#         #             for sequence_folder in os.listdir(building_path):
#         #                 image_path = os.path.join(building_path, sequence_folder, "step0.camera.png")
#         #                 if os.path.exists(image_path):
#         #                     all_image_paths.append(image_path)
#         #                     all_texts.append(f"This is a building called {building}.")
#         #         building_bar.update(1)
#         with tqdm(total=len(buildings), desc="Processing buildings", unit="building") as building_bar:
#             for building in buildings:
#                 building_path = os.path.join(root_dir, building)
#                 if os.path.isdir(building_path):
#                     # Liste alle Ordner (die Sequenzen) im Gebäudeordner auf
#                     sequence_folders = [f for f in os.listdir(building_path) if os.path.isdir(os.path.join(building_path, f))]
#                     for sequence_folder in sequence_folders:
#                         image_path = os.path.join(building_path, sequence_folder, "step0.camera.png")
#                         if os.path.exists(image_path):
#                             all_image_paths.append(image_path)
#                             all_texts.append(f"This is a building called {building}.")
#                 building_bar.update(1)
        

#         # Filtere schwarze Bilder
#         print("Filtering black images...")
#         filtered_image_paths = []
#         filtered_texts = []
#         for img_path, text in tqdm(
#             zip(all_image_paths, all_texts),
#             total=len(all_image_paths),
#             desc="Filtering images",
#             unit="image",
#         ):
#             #if not self.is_black_image(img_path):
#                 filtered_image_paths.append(img_path)
#                 filtered_texts.append(text)

#         # Begrenze die Anzahl der Daten, falls ein Limit angegeben ist
#         if limit is not None:
#             indices = random.sample(range(len(filtered_image_paths)), min(limit, len(filtered_image_paths)))
#             self.image_paths = [filtered_image_paths[i] for i in indices]
#             self.texts = [filtered_texts[i] for i in indices]
#         else:
#             self.image_paths = filtered_image_paths
#             self.texts = filtered_texts
#     def __len__(self):
#         """
#         Gibt die Anzahl der verfügbaren Datenpunkte zurück.
#         """
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         """
#         Gibt das Bild und den zugehörigen Text zurück.
#         """
#         image = Image.open(self.image_paths[idx]).convert("RGB")
#         text = clip.tokenize(self.texts[idx])[0]  # Tokenisiere Text

#         if self.transform:
#             image = self.transform(image)

#         return image, text

class BuildingDataset(Dataset):
    def __init__(self, root_dir, buildings, transform=None, black_threshold=10, limit=None):
        """
        :param root_dir: Wurzelverzeichnis der Daten
        :param buildings: Liste der zu ladenden Gebäude
        :param transform: Transformationen für die Bilder
        :param black_threshold: Schwelle für das Filtern schwarzer Bilder (Helligkeit)
        :param limit: Maximale Anzahl von Daten (zufällig ausgewählt)
        """
        self.image_paths = []
        self.texts = []
        self.tokenized_texts = []  # Liste für die tokenisierten Texte
        self.transform = transform
        self.buildings = buildings
        self.black_threshold = black_threshold

        print("Scanning directories and filtering images...")
        all_image_paths = []
        all_texts = []

        # Bilder und Texte sammeln
        with tqdm(total=len(buildings), desc="Processing buildings", unit="building") as building_bar:
            for building in buildings:
                building_path = os.path.join(root_dir, building)
                if os.path.isdir(building_path):
                    # Liste alle Ordner (die Sequenzen) im Gebäudeordner auf
                    sequence_folders = [f for f in os.listdir(building_path) if os.path.isdir(os.path.join(building_path, f))]
                    for sequence_folder in sequence_folders:
                        image_path = os.path.join(building_path, sequence_folder, "step0.camera.png")
                        if os.path.exists(image_path):
                            all_image_paths.append(image_path)
                            #all_texts.append(f"This is a building called {building}.")
                            all_texts.append(f"{building}")
                building_bar.update(1)

        # Tokenisiere die Texte einmalig und speichere sie
        self.tokenized_texts = clip.tokenize(all_texts).cpu()  # Tokenisieren aller Texte im Voraus
        self.image_paths = all_image_paths

        # Begrenze die Anzahl der Daten, falls ein Limit angegeben ist
        if limit is not None:
            indices = random.sample(range(len(self.image_paths)), min(limit, len(self.image_paths)))
            self.image_paths = [self.image_paths[i] for i in indices]
            self.tokenized_texts = [self.tokenized_texts[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Lade das Bild
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Verwende die bereits tokenisierten Texte
        text = self.tokenized_texts[idx]

        # Führe Transformationen auf das Bild durch, falls nötig
        if self.transform:
            image = self.transform(image)

        return image, text
    
def plot_images_with_labels(dataset, num_images=20):
    """
    Plotte eine bestimmte Anzahl von Bildern mit ihren Labels aus einem Dataset.

    :param dataset: Das Dataset, aus dem die Bilder und Labels geladen werden
    :param num_images: Anzahl der zu plottenden Bilder
    """
    num_images = min(num_images, len(dataset))  # Sicherstellen, dass wir nicht mehr als das Dataset haben
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 4x5 Grid für 20 Bilder
    axes = axes.flatten()

    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()  # Ändere die Dimensionen für matplotlib (HWC-Format)

        # Zeige das Bild und den zugehörigen Text
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")
        print(label)
        axes[i].axis('off')  # Deaktiviere Achsenbeschriftung

    # Blende überschüssige Subplots aus, falls weniger als 20 Bilder angezeigt werden
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def count_images_in_folder(folder_path, image_extensions=None):
    """
    Zählt die Anzahl der Bilder in einem Ordner.
    
    :param folder_path: Pfad zum Ordner
    :param image_extensions: Liste der erlaubten Bildformate (z. B. ['.jpg', '.png'])
    :return: Anzahl der Bilder im Ordner
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    # Konvertiere Erweiterungen in Kleinbuchstaben
    image_extensions = [ext.lower() for ext in image_extensions]

    # Zähle die Dateien, die eine der Erweiterungen haben
    image_count = sum(
        1 for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file)) and os.path.splitext(file)[1].lower() in image_extensions
    )
    return image_count

# use your own data
list_image_path = []
list_txt = []
buildings = ['A-Building', 'B-Building', 'C-Building', 'E-Building', 'F-Building',
                 'G-Building', 'H-Building', 'I-Building', 'L-Building', 'M-Building',
                 'N-Building', 'O-Building', 'R-Building', 'Z-Building']
for idx in range(count_images_in_folder("F:\\Studium\\Master\\Thesis\\Unity\\Furtwangen\\Recordings_Single\\")):
    # Gebäudeindex berechnen
    building_idx = idx // 12  # Wechselt nach je 12 Bildern das Gebäude
    number_idx = idx % 12  # Zahlen von 0000 bis 0011

    # Sorge dafür, dass der index nicht über die Anzahl der Gebäude hinaus geht
    if building_idx >= len(buildings):
        break

    # Generiere den Bildnamen basierend auf dem Gebäude und der laufenden Nummer
    building = buildings[building_idx]
    img_path = f"{image_path}Fuwa_{building}_{number_idx:04d}.png"  # Format 0000 bis 0011
    list_image_path.append(img_path)

    # Füge die ersten 120 Zeichen der Beschreibung hinzu
    list_txt.append(building)


dataset = BuildingDataset(image_path, buildings=buildings, transform=preprocess, limit=14000)
#dataset = image_title_dataset(list_image_path, list_txt)

train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True) #Define your own dataloader
#plot_images_with_labels(dataset, num_images=20)
# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model
num_epochs = 2
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
    save_directory = "./trained_clip_model"

    # Speichere das trainierte Modell
    torch.save(model.state_dict(), "best.pth")