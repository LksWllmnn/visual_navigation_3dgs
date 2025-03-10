# from PIL import Image

# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import os
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

# import clip
# from transformers import CLIPProcessor, CLIPModel

# # Load the CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


# # Choose computation device
# device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# # Load pre-trained CLIP model
# model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

# # Definiere eine benutzerdefinierte Dataset-Klasse, die das Label aus dem Bildnamen extrahiert
# class image_title_dataset(Dataset):
#     def __init__(self, list_image_path):
#         self.image_path = list_image_path
        
#     def __len__(self):
#         return len(self.image_path)

#     def __getitem__(self, idx):
#         # Bildpfad abrufen
#         img_path = self.image_path[idx]
#         # Extrahiere den Label aus dem Bildnamen, angenommen das Format ist 'fuwa_[label]_[Frame].png'
#         filename = os.path.basename(img_path)  # Z.B. 'fuwa_A-Buidling_0000.png'
#         label = filename.split('_')[1]  # Extrahiert 'A-Buidling' als Label
        
#         # Bild öffnen
#         image = Image.open(img_path).convert("RGB")
        
#         # Bild wird mit der CLIP-Vorverarbeitung bearbeitet
#         image = preprocess(image)
        
#         # Das Label wird tokenisiert (oder es könnte auch als Index verwendet werden)
#         # Hier wird der Titel (Label) als Tokenisiert bereitgestellt
#         tokenized_label = clip.tokenize([label])  # Tokenisierung des Labels
#         return image, tokenized_label[0]  # Gebe Bild und Label zurück

# # Beispiel für die Bildpfade (du solltest den Pfad zu deinen Bildern hier setzen)
# #image_path = r"F:\Studium\Master\Thesis\Unity\Furtwangen\Recordings_Single"
# image_path = r"C:\Users\Lukas\AppData\LocalLow\DefaultCompany\Fuwa_HDRP\Recordings_Single"
# list_image_path = []

# # Füge alle Bildpfade zu list_image_path hinzu
# for root, dirs, files in os.walk(image_path):
#     for file in files:
#         if file.endswith(".png"):  # Sicherstellen, dass nur .png-Dateien genommen werden
#             list_image_path.append(os.path.join(root, file))

# # Erstelle das Dataset und den DataLoader
# dataset = image_title_dataset(list_image_path)
# train_dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# # dataset = image_title_dataset(list_image_path, list_txt)
# # train_dataloader = DataLoader(dataset, batch_size=1000, shuffle=True) #Define your own dataloader

# # Function to convert model's parameters to FP32 format
# def convert_models_to_fp32(model): 
#     for p in model.parameters(): 
#         p.data = p.data.float() 
#         p.grad.data = p.grad.data.float() 


# if device == "cpu":
#   model.float()

# # Prepare the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# # Specify the loss function
# loss_img = nn.CrossEntropyLoss()
# loss_txt = nn.CrossEntropyLoss()

# # Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     pbar = tqdm(train_dataloader, total=len(train_dataloader))
#     for batch in pbar:
#         optimizer.zero_grad()

#         images,texts = batch 
        
#         images= images.to(device)
#         texts = texts.to(device)

#         # Forward pass
#         logits_per_image, logits_per_text = model(images, texts)

#         # Compute loss
#         ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
#         total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

#         # Backward pass
#         total_loss.backward()
#         if device == "cpu":
#             optimizer.step()
#         else : 
#             convert_models_to_fp32(model)
#             optimizer.step()
#             clip.model.convert_weights(model)

#         pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

#     # Speichere das trainierte Modell
#     torch.save(model.state_dict(), "best_2.pt")



# import os
# from PIL import Image
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import clip
# from transformers import CLIPProcessor, CLIPModel
# import random

# # Load the CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# # Choose computation device
# device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# # Load pre-trained CLIP model
# model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

# # Define a custom Dataset class that extracts the label from the image name
# class ImageTitleDataset(Dataset):
#     def __init__(self, list_image_path, limit_data=False, limit_count=1000):
#         self.image_path = list_image_path
#         if limit_data:
#             random.shuffle(self.image_path)  # Shuffle the list of images randomly
#             self.image_path = self.image_path[:limit_count]  # Limit to the first 'limit_count' images after shuffle

#     def __len__(self):
#         return len(self.image_path)

#     def __getitem__(self, idx):
#         # Get image path
#         img_path = self.image_path[idx]
#         # Extract the label from the image filename (assuming format 'fuwa_[label]_[Frame].png')
#         filename = os.path.basename(img_path)  # e.g., 'fuwa_A-Building_0000.png'
#         label = filename.split('_')[1]  # Extract 'A-Building' as the label
        
#         # Open the image
#         image = Image.open(img_path).convert("RGB")
        
#         # Preprocess the image with CLIP's preprocessing pipeline
#         image = preprocess(image)
        
#         # Tokenize the label (or you can use it as an index)
#         tokenized_label = clip.tokenize([label])  # Tokenize the label
#         return image, tokenized_label[0]  # Return the image and label

# # Set the path to your image directory
# #image_path = r"F:\Studium\Master\Thesis\Unity\Furtwangen\Recordings_Single"
# image_path = r"C:\Users\Lukas\AppData\LocalLow\DefaultCompany\Fuwa_HDRP\Recordings_Single"
# list_image_path = []

# # Add all image paths to list_image_path
# for root, dirs, files in os.walk(image_path):
#     for file in files:
#         if file.endswith(".png"):  # Ensure that only .png files are included
#             list_image_path.append(os.path.join(root, file))

# # Flag to limit the data
# limit_data = True  # Set this to True to limit the dataset size
# limit_count = 1000  # Number of images to use if limit_data is True

# # Create the dataset and DataLoader
# dataset = ImageTitleDataset(list_image_path, limit_data=limit_data, limit_count=limit_count)
# train_dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# # Function to convert model's parameters to FP32 format
# def convert_models_to_fp32(model): 
#     for p in model.parameters(): 
#         p.data = p.data.float() 
#         p.grad.data = p.grad.data.float() 

# if device == "cpu":
#     model.float()

# # Prepare the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)  # Use a small learning rate for fine-tuning

# # Specify the loss function
# loss_img = nn.CrossEntropyLoss()
# loss_txt = nn.CrossEntropyLoss()

# # Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     pbar = tqdm(train_dataloader, total=len(train_dataloader))
#     for batch in pbar:
#         optimizer.zero_grad()

#         images, texts = batch 
        
#         images = images.to(device)
#         texts = texts.to(device)

#         # Forward pass
#         logits_per_image, logits_per_text = model(images, texts)

#         # Compute loss
#         ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
#         total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

#         # Backward pass
#         total_loss.backward()
#         if device == "cpu":
#             optimizer.step()
#         else: 
#             convert_models_to_fp32(model)
#             optimizer.step()
#             clip.model.convert_weights(model)

#         pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

#     # Save the trained model
#     torch.save(model.state_dict(), "best_tall_1.pt")


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

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

# Define a custom Dataset class that extracts the label from the image name
class ImageTitleDataset(Dataset):
    def __init__(self, list_image_path, limit_data=False, limit_count=1000):
        self.image_path = list_image_path
        if limit_data:
            random.shuffle(self.image_path)  # Shuffle the list of images randomly
            self.image_path = self.image_path[:limit_count]  # Limit to the first 'limit_count' images after shuffle

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # Get image path
        img_path = self.image_path[idx]
        # Extract the label from the image filename (assuming format 'fuwa_[label]_[Frame].png')
        filename = os.path.basename(img_path)  # e.g., 'fuwa_A-Building_0000.png'
        label = filename.split('_')[1]  # Extract 'A-Building' as the label
        #label_key = filename.split('_')[1]  # Extract 'A-Building'

        # Define the mapping for the switch functionality
        # label_mapping = {
        #     "A-Building": "Albert",
        #     "B-Building": "Bertram",
        #     "C-Building": "Cilian",
        #     "E-Building": "Ezachiel",
        #     "F-Building": "Florian",
        #     "G-Building": "Gustav",
        #     "H-Building": "Herbert",
        #     "I-Building": "Ignazius",
        #     "L-Building": "Lasse",
        #     "M-Building": "Michael",
        #     "N-Building": "Nicklas",
        #     "O-Building": "Oskar",
        #     "R-Building": "Raphael",
        #     "Z-Building": "Zachariah"
        # }
        # translations = {
        #     "A-Building": ["A-Building", "Bâtiment-A", "Edificio-A", "A-Gebäude", "A-Byggnad", "A-gebouw", "Budynek-A", "A-Binası", "A-Biru", "A-Bâtiment"],
        #     "B-Building": ["B-Building", "Bâtiment-B", "Edificio-B", "B-Gebäude", "B-Byggnad", "B-gebouw", "Budynek-B", "B-Binası", "B-Biru", "B-Bâtiment"],
        #     "C-Building": ["C-Building", "Bâtiment-C", "Edificio-C", "C-Gebäude", "C-Byggnad", "C-gebouw", "Budynek-C", "C-Binası", "C-Biru", "C-Bâtiment"],
        #     "E-Building": ["E-Building", "Bâtiment-E", "Edificio-E", "E-Gebäude", "E-Byggnad", "E-gebouw", "Budynek-E", "E-Binası", "E-Biru", "E-Bâtiment"],
        #     "F-Building": ["F-Building", "Bâtiment-F", "Edificio-F", "F-Gebäude", "F-Byggnad", "F-gebouw", "Budynek-F", "F-Binası", "F-Biru", "F-Bâtiment"],
        #     "G-Building": ["G-Building", "Bâtiment-G", "Edificio-G", "G-Gebäude", "G-Byggnad", "G-gebouw", "Budynek-G", "G-Binası", "G-Biru", "G-Bâtiment"],
        #     "H-Building": ["H-Building", "Bâtiment-H", "Edificio-H", "H-Gebäude", "H-Byggnad", "H-gebouw", "Budynek-H", "H-Binası", "H-Biru", "H-Bâtiment"],
        #     "I-Building": ["I-Building", "Bâtiment-I", "Edificio-I", "I-Gebäude", "I-Byggnad", "I-gebouw", "Budynek-I", "I-Binası", "I-Biru", "I-Bâtiment"],
        #     "L-Building": ["L-Building", "Bâtiment-L", "Edificio-L", "L-Gebäude", "L-Byggnad", "L-gebouw", "Budynek-L", "L-Binası", "L-Biru", "L-Bâtiment"],
        #     "M-Building": ["M-Building", "Bâtiment-M", "Edificio-M", "M-Gebäude", "M-Byggnad", "M-gebouw", "Budynek-M", "M-Binası", "M-Biru", "M-Bâtiment"],
        #     "N-Building": ["N-Building", "Bâtiment-N", "Edificio-N", "N-Gebäude", "N-Byggnad", "N-gebouw", "Budynek-N", "N-Binası", "N-Biru", "N-Bâtiment"],
        #     "O-Building": ["O-Building", "Bâtiment-O", "Edificio-O", "O-Gebäude", "O-Byggnad", "O-gebouw", "Budynek-O", "O-Binası", "O-Biru", "O-Bâtiment"],
        #     "R-Building": ["R-Building", "Bâtiment-R", "Edificio-R", "R-Gebäude", "R-Byggnad", "R-gebouw", "Budynek-R", "R-Binası", "R-Biru", "R-Bâtiment"],
        #     "Z-Building": ["Z-Building", "Bâtiment-Z", "Edificio-Z", "Z-Gebäude", "Z-Byggnad", "Z-gebouw", "Budynek-Z", "Z-Binası", "Z-Biru", "Z-Bâtiment"]
        # }


        # Zufällige Reihenfolge der Übersetzungen generieren
        # label_mapping = {}
        # for building, translation_list in translations.items():
        #     random.shuffle(translation_list)
        #     label_mapping[building] = " ".join(translation_list)

        # # Ausgabe der generierten Labels
        # label = label_mapping.get(label_key, "Unknown")
        
        
        # Open the image
        image = Image.open(img_path).convert("RGB")
        
        # Preprocess the image with CLIP's preprocessing pipeline
        image = preprocess(image)
        
        # Tokenize the label (or you can use it as an index)
        tokenized_label = clip.tokenize([label])  # Tokenize the label
        return image, tokenized_label[0]  # Return the image and label

# Set the path to your image directory
image_path = r"F:\Studium\Master\Thesis\data\singleBuildings_Resnet2\Recordings_Single"
#image_path = r"F:\Studium\Master\Thesis\Unity\Furtwangen\Recordings_Single"
list_image_path = []

# Add all image paths to list_image_path
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith(".png"):  # Ensure that only .png files are included
            list_image_path.append(os.path.join(root, file))

# Flag to limit the data
limit_data = False  # Set this to True to limit the dataset size
limit_count = 3000  # Number of images to use if limit_data is True

if limit_data:
    # random.shuffle(list_image_path)  # Shuffle the list of images if limiting data
    # sample_images = list_image_path[:limit_count]  # Use the first 'limit_count' images
    # Gruppiere Bilder nach Gebäudetyp
    grouped_images = defaultdict(list)
    for img_path in tqdm(list_image_path, desc="Grouping images by building type"):
        label = os.path.basename(img_path).split('_')[1]  # Extrahiere Gebäudetyp
        grouped_images[label].append(img_path)

    # Gleiche Anzahl von Bildern aus jeder Gruppe auswählen
    balanced_images = []
    num_images_per_building = limit_count // len(grouped_images)  # Anzahl Bilder pro Gebäudetyp
    for label, images in tqdm(grouped_images.items(), desc="Selecting balanced images"):
        random.shuffle(images)  # Bilder innerhalb jeder Gruppe mischen
        balanced_images.extend(images[:num_images_per_building])  # Füge eine begrenzte Anzahl hinzu

    # Wenn noch Platz ist, fülle die Auswahl mit den restlichen Bildern
    remaining_count = limit_count - len(balanced_images)
    if remaining_count > 0:
        leftover_images = [
            img for images in tqdm(grouped_images.values(), desc="Finding leftover images") 
            for img in images if img not in balanced_images
        ]
        random.shuffle(leftover_images)
        balanced_images.extend(leftover_images[:remaining_count])

    # Verwende die ausgewählten Bilder
    sample_images = balanced_images
else:
    sample_images = list_image_path  # Use all images

# Plot 20 random images with labels
plt.figure(figsize=(15, 10))
for i in range(20):
    img_path = sample_images[i]
    filename = os.path.basename(img_path)
    label = filename.split('_')[1]  # Extract 'A-Building' as the label
    image = Image.open(img_path).convert("RGB")
    ax = plt.subplot(4, 5, i + 1)  # Create a subplot
    ax.imshow(image)
    ax.set_title(label)  # Set the label as the title of the image
    ax.axis('off')  # Hide axis

plt.tight_layout()
plt.show()

# Create the dataset and DataLoader
dataset = ImageTitleDataset(list_image_path, limit_data=limit_data, limit_count=limit_count)
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

# # Train the model
# num_epochs = 40
# for epoch in range(num_epochs):
#     pbar = tqdm(train_dataloader, total=len(train_dataloader))
#     for batch in pbar:
#         optimizer.zero_grad()

#         images, texts = batch 
        
#         images = images.to(device)
#         texts = texts.to(device)

#         # Forward pass
#         logits_per_image, logits_per_text = model(images, texts)

#         # Compute loss
#         ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
#         total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

#         # Backward pass
#         total_loss.backward()
#         if device == "cpu":
#             optimizer.step()
#         else: 
#             convert_models_to_fp32(model)
#             optimizer.step()
#             clip.model.convert_weights(model)

#         pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

# Initialize an empty list to store epoch-wise average losses
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
    torch.save(model.state_dict(), "clip_tall_20b_1e6lr_40e_2.pth")

# Plot the average loss per epoch
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), epoch_losses, marker='o', label='Average Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Average Loss per Epoch")
plt.legend()
plt.grid()
plt.show()

    
