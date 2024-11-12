import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import torch.nn.functional as F
from custom_cnn import CustomCNN

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Lade dein trainiertes CNN-Modell
    model = CustomCNN(num_classes=15)  # Passe die Anzahl der Klassen an
    model.load_state_dict(torch.load("./chkpt/custom_cnn_1.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Setze das Modell in den Evaluationsmodus

    classes = ["A-Building", "B-Building", "C-Building", "E-Building", "F-Building", "G-Building", "H-Building", 
               "I-Building", "J-Building", "L-Building", "M-Building", "N-Building", "O-Building", "R-Building", "Z-Building"]

    # Anpassung der Bildtransformationen
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    custom_images_path = "./output_masks/"
    custom_dataset = CustomImageDataset(image_folder=custom_images_path, transform=preprocess)

    loader = DataLoader(custom_dataset, batch_size=32, shuffle=False, num_workers=2)

    confidence_threshold = 50  # Konfidenz-Schwelle in Prozent

    with torch.no_grad():
        for images, img_paths in loader:
            images = images.to(device)

            # Verwende dein trainiertes CNN-Modell für die Vorhersage
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)  # Wahrscheinlichkeiten berechnen

            top_probs, top_labels = probs.topk(1, dim=1)

            for img_path, top_label, top_prob in zip(img_paths, top_labels, top_probs):
                confidence = top_prob.item() * 100  # Umrechnung in Prozent
                if confidence < confidence_threshold:
                    # Lösche das Bild, wenn die Konfidenz unter der Schwelle liegt
                    print(f"Deleting {img_path} due to low confidence: {confidence:.2f}%")
                    os.remove(img_path)
                else:
                    predicted_class = classes[top_label.item()]
                    print(f"Image: {img_path} -> Predicted Class: {predicted_class} with confidence: {confidence:.2f}%")
