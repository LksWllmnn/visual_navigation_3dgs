import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((28, 28))  # Output size of (28, 28) for larger input images
        self.fc1 = nn.Linear(64 * 28 * 28, 512)  # Adjusted input size for the fully connected layer
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)  # Adaptive pooling to a fixed size
        x = x.view(-1, 64 * 28 * 28)  # Flatten the tensor for larger images
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        self.class_mapping = {
            "A-Building": 0,
            "B-Building": 1,
            "C-Building": 2,
            "E-Building": 3,
            "F-Building": 4,
            "G-Building": 5,
            "H-Building": 6,
            "I-Building": 7,
            "J-Building": 8,
            "L-Building": 9,
            "M-Building": 10,
            "N-Building": 11,
            "O-Building": 12,
            "R-Building": 13,
            "Z-Building": 14,
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)  # Load image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Extrahiere den Dateinamen und die Klasse
        filename = os.path.basename(img_path)
        label = self.extract_label(filename)  # Extrahiere das Label

        # Konvertiere Label in Index
        label_index = self.class_mapping.get(label, -1)  # Standardwert -1, wenn nicht gefunden

        if self.transform:
            image = self.transform(image)

        return image, label_index

    def extract_label(self, filename):
        # Zerlege den Dateinamen und extrahiere das Label
        parts = filename.split('_')
        if len(parts) >= 2:  # Sicherstellen, dass genügend Teile vorhanden sind
            return parts[1]  # Das Label ist der zweite Teil des Dateinamens
        return -1  # Rückgabe eines Standardwerts, wenn das Label nicht gefunden wurde

# Daten laden und Transformationsoperationen definieren
def load_data(image_folder, batch_size=8):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert from numpy array to PIL Image
        transforms.Resize((1024, 1024)),  # Larger resize to 1024x1024
        transforms.ToTensor(),  # Convert to Tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    dataset = CustomImageDataset(image_folder=image_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()  # Move to GPU if available

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    # Setze den Pfad zu deinen Bildern
    image_folder = "./data/single_building_images"  # Pfad zu deinem Bildordner
    batch_size = 8  # Batch-Größe
    num_epochs = 20  # Anzahl der Trainingsiterationen (Epochs)

    dataloader = load_data(image_folder, batch_size=batch_size)

    # Modell initialisieren
    model = CustomCNN(num_classes=15).cuda()  # GPU verwenden, falls verfügbar

    # Trainiere das Modell
    train_model(model, dataloader, num_epochs=num_epochs, learning_rate=0.001)

    torch.save(model.state_dict(), './chkpt/custom_cnn_1.pth')
