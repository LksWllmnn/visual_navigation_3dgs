from PIL import Image
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import clip

# Dataset für deine Ordnerstruktur
class BuildingDataset(Dataset):
    def __init__(self, root_dir, buildings, transform=None, black_threshold=10):
        """
        :param root_dir: Wurzelverzeichnis der Daten
        :param buildings: Liste der zu ladenden Gebäude
        :param transform: Transformationen für die Bilder
        :param black_threshold: Schwelle für das Filtern schwarzer Bilder (Helligkeit)
        """
        self.image_paths = []
        self.texts = []
        self.transform = transform
        self.buildings = buildings
        self.black_threshold = black_threshold

        print("Scanning directories and filtering images...")
        all_image_paths = []
        all_texts = []

        # Fortschrittsbalken für Gebäude
        with tqdm(total=len(buildings), desc="Processing buildings", unit="building") as building_bar:
            for building in buildings:
                building_path = os.path.join(root_dir, building)
                if os.path.isdir(building_path):
                    for sequence_folder in os.listdir(building_path):
                        image_path = os.path.join(building_path, sequence_folder, "step0.camera.png")
                        if os.path.exists(image_path):
                            all_image_paths.append(image_path)
                            all_texts.append(f"This is a building called {building}.")
                building_bar.update(1)

        # Filtere schwarze Bilder
        print("Filtering black images...")
        for img_path, text in tqdm(
            zip(all_image_paths, all_texts),
            total=len(all_image_paths),
            desc="Filtering images",
            unit="image",
        ):
            if not self.is_black_image(img_path):
                self.image_paths.append(img_path)
                self.texts.append(text)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Gibt das Bild und den zugehörigen Text zurück.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = clip.tokenize(self.texts[idx])[0]  # Tokenisiere Text

        if self.transform:
            image = self.transform(image)

        return image, text

    def is_black_image(self, img_path):
        """
        Prüft, ob ein Bild fast vollständig schwarz ist.
        :param img_path: Pfad zum Bild
        :return: True, wenn das Bild schwarz ist, sonst False
        """
        with Image.open(img_path) as img:
            img = img.convert("L")  # In Graustufen konvertieren
            pixel_values = list(img.getdata())
            avg_pixel_value = sum(pixel_values) / len(pixel_values)
            return avg_pixel_value < self.black_threshold


# Funktion, um Modellparameter für CPU auf FP32 umzuwandeln
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


# Hauptfunktion für Training
def train_clip(root_dir, buildings, num_epochs=40, batch_size=25, save_path="trained_clip_model.pth"):
    # Lade das CLIP-Modell und die Vorverarbeitungsfunktion
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

    # Erstelle Dataset und DataLoader
    dataset = BuildingDataset(root_dir=root_dir, buildings=buildings, transform=preprocess)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modell vorbereiten
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Training mit Fortschrittsanzeige
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", leave=True)

        for batch in pbar:
            optimizer.zero_grad()

            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)

            # Vorwärtsdurchlauf
            logits_per_image, logits_per_text = model(images, texts)

            # Verlust berechnen
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            # Rückwärtsdurchlauf und Schritt des Optimierers
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            # Fortschrittsbalken aktualisieren
            pbar.set_postfix(loss=total_loss.item())

    # Speichere das trainierte Modell
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Pfade und Parameter
    root_dir = "C:/Users/Lukas/AppData/LocalLow/DefaultCompany/Fuwa_HDRP/singleBuildings_ResnetSam"  # Root-Verzeichnis der Gebäudedaten
    buildings = ['A-Building', 'B-Building', 'C-Building', 'E-Building', 'F-Building',
                 'G-Building', 'H-Building', 'I-Building', 'L-Building', 'M-Building',
                 'N-Building', 'O-Building', 'R-Building', 'Z-Building']
    num_epochs = 40
    batch_size = 50
    save_path = "trained_clip_model.pth"

    # Training starten
    train_clip(root_dir=root_dir, buildings=buildings, num_epochs=num_epochs, batch_size=batch_size, save_path=save_path)
