import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import random  # Für das Mischen der Daten

# class CustomImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None, max_samples=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []

#         for label_folder in os.listdir(root_dir):
#             label_path = os.path.join(root_dir, label_folder)
#             if os.path.isdir(label_path):
#                 for sequence_folder in os.listdir(label_path):
#                     sequence_path = os.path.join(label_path, sequence_folder)
#                     if os.path.isdir(sequence_path):
#                         image_path = os.path.join(sequence_path, "step0.camera.png")
#                         if os.path.exists(image_path):
#                             self.image_paths.append(image_path)
#                             self.labels.append(label_folder)

#         # Datenmischung und Begrenzung der Anzahl
#         combined = list(zip(self.image_paths, self.labels))
#         random.shuffle(combined)  # Mischen der Daten
#         self.image_paths, self.labels = zip(*combined)

#         if max_samples is not None:
#             self.image_paths = self.image_paths[:max_samples]
#             self.labels = self.labels[:max_samples]

#         self.label_encoder = LabelEncoder()
#         self.labels = self.label_encoder.fit_transform(self.labels)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label

from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None, black_threshold=10):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.black_threshold = black_threshold

        print("Scanning directories and filtering images...")
        # Fortschrittsbalken für Verzeichnisse und Bilder
        all_image_paths = []
        all_labels = []

        label_folders = os.listdir(root_dir)
        with tqdm(total=len(label_folders), desc="Processing label folders", unit="folder") as folder_bar:
            for label_folder in label_folders:
                label_path = os.path.join(root_dir, label_folder)
                if os.path.isdir(label_path):
                    for sequence_folder in os.listdir(label_path):
                        sequence_path = os.path.join(label_path, sequence_folder)
                        if os.path.isdir(sequence_path):
                            image_path = os.path.join(sequence_path, "step0.camera.png")
                            if os.path.exists(image_path):
                                all_image_paths.append(image_path)
                                all_labels.append(label_folder)
                folder_bar.update(1)

        # Filtere schwarze Bilder
        print("Filtering black images...")
        for img_path, label in tqdm(zip(all_image_paths, all_labels), total=len(all_image_paths), desc="Filtering images", unit="image"):
            if not self.is_black_image(img_path):
                self.image_paths.append(img_path)
                self.labels.append(label)

        # Datenmischung und Begrenzung der Anzahl
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)  # Mischen der Daten
        self.image_paths, self.labels = zip(*combined)

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
            self.labels = self.labels[:max_samples]

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

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

def show_images(dataloader, dataset):
    images, labels = next(iter(dataloader))
    images = images.permute(0, 2, 3, 1)
    text_labels = dataset.label_encoder.inverse_transform(labels.numpy())
    plt.figure(figsize=(12, 6))
    for i in range(len(images)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(text_labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10):
#     since = time.time()
#     best_model_wts = model.state_dict()
#     best_acc = 0.0

#     train_loss_history = []
#     val_loss_history = []
#     train_acc_history = []
#     val_acc_history = []

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             if phase == 'train':
#                 train_loss_history.append(epoch_loss)
#                 train_acc_history.append(epoch_acc.item())
#             else:
#                 val_loss_history.append(epoch_loss)
#                 val_acc_history.append(epoch_acc.item())

#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = model.state_dict()

#         print()

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:.4f}')

#     model.load_state_dict(best_model_wts)

#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss_history, label='Train Loss')
#     plt.plot(val_loss_history, label='Val Loss')
#     plt.title('Loss over epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(train_acc_history, label='Train Accuracy')
#     plt.plot(val_acc_history, label='Val Accuracy')
#     plt.title('Accuracy over epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     return model

from tqdm import tqdm  # Für Fortschrittsbalken

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Fortschrittsbalken für die Batches
            progress_bar = tqdm(
                dataloaders[phase],
                desc=f'{phase.capitalize()} Phase Progress',
                leave=True,
                unit='batch'
            )

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Fortschrittsanzeige aktualisieren
                current_loss = running_loss / (len(progress_bar) * inputs.size(0))
                progress_bar.set_postfix(
                    loss=current_loss,
                    acc=(running_corrects.double() / (len(progress_bar) * inputs.size(0))).item()
                )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model


def main():
    # Dataset Pfad anpassen
    image_path = "C:/Users/Lukas/AppData/LocalLow/DefaultCompany/Fuwa_HDRP/singleBuildings_ResnetSam"  # Hier den Pfad zur Wurzel der Ordnerstruktur angeben

    # Maximal erlaubte Datenanzahl
    max_samples = 14000  # Hier die Obergrenze festlegen

    # Gerät festlegen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformationen definieren
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    # Dataset erstellen
    dataset = CustomImageDataset(root_dir=image_path, transform=data_transforms['train'], max_samples=max_samples)

    # Dataset aufteilen
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # Dataloader erstellen
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False)
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }

    # Beispielbilder anzeigen
    show_images(dataloaders['train'], train_dataset.dataset)

    # Vortrainiertes ResNet50-Modell laden
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    label_mapping = dataset.label_encoder.classes_
    model_ft.fc = nn.Linear(num_ftrs, len(label_mapping))
    model_ft = model_ft.to(device)

    # Verlustfunktion und Optimierer definieren
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Modell trainieren
    model_ft = train_model(
        model=model_ft,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer_ft,
        device=device,
        num_epochs=15
    )

    # Modell speichern
    torch.save(model_ft.state_dict(), "resnet50_finetuned.pth")

if __name__ == "__main__":
    main()
