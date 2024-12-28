import os
import torch
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Initialisierung des Geräts
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Laden des CLIP-Modells und der Vorverarbeitungsfunktionen
def load_clip_model():
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
    return model, preprocess

# Parameter des Modells ausgeben
def print_model_info(model):
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

# Zeroshot-Klassifikator erstellen
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # Formatieren mit der Klasse
            texts = clip.tokenize(texts).cuda()  # Tokenisieren
            class_embeddings = model.encode_text(texts)  # Embedding mit dem Text-Encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

# Genauigkeit berechnen
def calculate_accuracy(img_paths, top_labels, imagenet_classes):
    correct = 0
    total = len(img_paths)

    for img_path, top_label in zip(img_paths, top_labels):
        # Extrahiere den Gebäudenamen aus dem Dateipfad
        img_filename = os.path.basename(img_path)
        building_name_from_path = img_filename.split('_')[1]  # Extrahiert den Gebäudenamen

        # Vorhergesagter Klassenname
        predicted_class = imagenet_classes[top_label.item()]

        # Überprüfen, ob der vorhergesagte Name mit dem extrahierten Namen übereinstimmt
        if building_name_from_path in predicted_class:
            correct += 1

    # Berechne die Genauigkeit in Prozent
    accuracy = (correct / total) * 100
    return accuracy

# Dataset-Klasse für benutzerdefinierte Bilder
class CustomImageDataset(Dataset):
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
        return image, img_path  # Bild und Pfad zurückgeben

# Laden der benutzerdefinierten Bilder und Vorverarbeitung
def load_custom_images(image_folder, preprocess):
    custom_dataset = CustomImageDataset(image_folder=image_folder, transform=preprocess)
    loader = DataLoader(custom_dataset, batch_size=32, shuffle=False, num_workers=2)
    return loader

# Hauptfunktion
def main():
    # Laden des CLIP-Modells
    model, preprocess = load_clip_model()

    # Modell laden
    # model.load_state_dict(torch.load("best_2.pt", map_location=device))
    model.load_state_dict(torch.load("clip_tall_20b_1e6lr_40e_wholeScene.pth", map_location=device))
    model.to(device)

    # Modellinformationen ausgeben
    print_model_info(model)

    # Die Bildklassen und Templates für die Zeroshot-Klassifikation
    imagenet_classes = ["A-Building", "B-Building", "C-Building", "E-Building", "F-Building", "G-Building", 
                        "H-Building", "I-Building", "L-Building", "M-Building", "N-Building", 
                        "O-Building", "R-Building", "Z-Building"]

    imagenet_templates = ["{}"]  # Dein Template-Array für die Klassifikationen

    # Zeroshot-Gewichte berechnen
    zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model)

    # Bilder aus benutzerdefiniertem Verzeichnis laden
    custom_images_path = r"F:\Studium\Master\Thesis\Unity\Furtwangen\Recordings_Single"
    loader = load_custom_images(custom_images_path, preprocess)

    # Ergebnislisten
    all_img_paths = []
    all_top_labels = []

    # Vorhersagen über die benutzerdefinierten Bilder
    with torch.no_grad():
        for images, img_paths in tqdm(loader):
            images = images.cuda()

            # Bilder-Features extrahieren
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Logits berechnen (Ähnlichkeit der Bild-Features mit den Text-Embeddings der Klassen)
            logits = 100. * image_features @ zeroshot_weights

            # Wahrscheinlichkeiten berechnen
            probs = logits.softmax(dim=-1)
            top_probs, top_labels = probs.topk(1, dim=-1)

            # Bildpfade und Labels speichern
            all_img_paths.extend(img_paths)
            all_top_labels.extend(top_labels.cpu())

            # Vorhergesagte Klasse für jedes Bild ausgeben
            for img_path, top_label in zip(img_paths, top_labels):
                predicted_class = imagenet_classes[top_label.item()]
                print(f"Image: {img_path} -> Predicted Class: {predicted_class}")

    # Genauigkeit berechnen
    accuracy = calculate_accuracy(all_img_paths, all_top_labels, imagenet_classes)
    print(f"Model accuracy: {accuracy:.2f}%")

# Skript ausführen
if __name__ == "__main__":
    main()
