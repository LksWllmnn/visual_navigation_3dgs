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
        
        # Open the image
        image = Image.open(img_path).convert("RGB")
        
        # Preprocess the image with CLIP's preprocessing pipeline
        image = preprocess(image)
        
        # Tokenize the label (or you can use it as an index)
        tokenized_label = clip.tokenize([label])  # Tokenize the label
        return image, tokenized_label[0]  # Return the image and label

# Set the path to your image directory
image_path = r"C:\Users\Lukas\AppData\LocalLow\DefaultCompany\Fuwa_HDRP\Recordings_Single"
list_image_path = []

# Add all image paths to list_image_path
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith(".png"):  # Ensure that only .png files are included
            list_image_path.append(os.path.join(root, file))

# Flag to limit the data
limit_data = True  # Set this to True to limit the dataset size
limit_count = 3000  # Number of images to use if limit_data is True

# If we want to plot images, let's select 20 random images
if limit_data:
    random.shuffle(list_image_path)  # Shuffle the list of images if limiting data
    sample_images = list_image_path[:limit_count]  # Use the first 'limit_count' images
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
train_dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)  # Use a small learning rate for fine-tuning

# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
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

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else: 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "best_tall_1.pt")
