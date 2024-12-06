from collections import defaultdict
import os
import random
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm  # Fortschrittsbalken hinzufügen
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import json
# from torchvision.models.detection.coco_eval import CocoEvaluator
# from torchvision.models.detection.coco_utils import get_coco_api_from_dataset

# Define the color-to-ID mapping
COLOR_TO_ID = {
    (0, 0, 0, 255): 0,  # Background
    (0, 0, 255, 255): 1,    # A-Building
    (0, 255, 0, 255): 2,    # B-Building
    (255, 0, 0, 255): 3,    # C-Building
    (255, 255, 255, 255): 4,  # E-Building
    (255, 235, 4, 255): 5,  # F-Building
    (128, 128, 128, 255): 6,  # G-Building
    (255, 32, 98, 255): 7,  # H-Building
    (255, 25, 171, 255): 8,  # I-Building
    (93, 71, 255, 255): 9,  # L-Building
    (255, 73, 101, 255): 10,  # M-Building
    (145, 255, 114, 255): 11,  # N-Building
    (153, 168, 255, 255): 12,  # O-Building
    (64, 0, 75, 255): 13,  # R-Building
    (18, 178, 0, 255): 14,  # Z-Building
    #(255, 169, 0, 255): 15,  # Other
}

class BuildingSegmentationDataset:
    def __init__(self, root, transforms=None, num_images_per_building=3000):
        self.root = root
        self.transforms = transforms
        self.num_images_per_building = num_images_per_building

        # Liste aller Gebäude abrufen
        self.buildings = sorted(os.listdir(root))
        self.imgs = []
        self.masks = []

        print("Lade Sequenzen...")

        # Durch alle Gebäude iterieren
        for building in tqdm(self.buildings, desc="Fortschritt", unit="building"):
            building_path = os.path.join(root, building)
            if os.path.isdir(building_path):
                # Liste der Sequenzen des Gebäudes
                sequence_folders = os.listdir(building_path)

                building_images = []
                building_masks = []

                for sequence_folder in sequence_folders:
                    # Wenn die Sequenz ein Ordner ist, gehen wir weiter
                    sequence_path = os.path.join(building_path, sequence_folder)
                    print(sequence_path)
                    if os.path.isdir(sequence_path):
                        # Direkt die Bild- und Maskenpfade definieren
                        img_file = os.path.join(sequence_path, "step0.camera.png")
                        mask_file = os.path.join(sequence_path, "step0.camera.semantic segmentation.png")
                        
                        print(f"Bildpfad: {img_file}")
                        print(f"Maskenpfad: {mask_file}")
                        
                        # Überprüfen, ob beide Dateien existieren
                        if os.path.exists(img_file) and os.path.exists(mask_file):
                            # Instanzen aus der JSON-Maske zählen
                            instance_counts = self.count_instances_in_json(mask_file)
                            
                            # Nur Bilder mit Instanzen hinzufügen
                            if instance_counts:
                                building_images.append(img_file)
                                building_masks.append(mask_file)

                # Wenn für ein Gebäude Bilder mit Instanzen gesammelt wurden
                if building_images:
                    self.imgs.extend(building_images)
                    self.masks.extend(building_masks)

                # Sicherstellen, dass wir nur eine bestimmte Anzahl von Bildern pro Gebäude haben
                if len(building_images) > self.num_images_per_building:
                    indices = random.sample(range(len(building_images)), self.num_images_per_building)
                    building_images = [building_images[i] for i in indices]
                    building_masks = [building_masks[i] for i in indices]

                self.imgs.extend(building_images)
                self.masks.extend(building_masks)
                
    def count_instances_in_json(self, json_file):
        instance_counts = defaultdict(int)
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "captures" in data:
                for capture in data["captures"]:
                    if "annotations" in capture:
                        for annotation in capture["annotations"]:
                            if annotation.get("@type") == "type.unity.com/unity.solo.SemanticSegmentationAnnotation":
                                instances = annotation.get("instances", [])
                                if instances:
                                    for instance in instances:
                                        label_name = instance.get("labelName")
                                        if label_name:
                                            instance_counts[label_name] += 1
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Fehler beim Laden der Datei {json_file}: {e}")
        
        return instance_counts

    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGBA")

        # Convert mask to class IDs
        mask_np = np.array(mask)
        instance_mask = np.zeros(mask_np.shape[:2], dtype=np.int64)
        for color, class_id in COLOR_TO_ID.items():
            instance_mask[np.all(mask_np == np.array(color), axis=-1)] = class_id

        # Get unique class IDs and generate binary masks
        obj_ids = np.unique(instance_mask)
        obj_ids = obj_ids[obj_ids > 0]  # Exclude background (0)

        if len(obj_ids) == 0:
            # No objects: set empty bounding boxes and labels
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, instance_mask.shape[0], instance_mask.shape[1]), dtype=torch.uint8)
        else:
            # Normal processing
            masks = instance_mask == obj_ids[:, None, None]
            boxes = []
            for i in range(len(obj_ids)):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(obj_ids, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Filter invalid bounding boxes
        if len(boxes) > 0:
            # Ensure positive height and width
            valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_indices]
            labels = labels[valid_indices]
            masks = masks[valid_indices]

        image_id = torch.tensor([idx])
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([0.0])  # Set area to 0 if no boxes exist
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    # def __init__(self, root, transforms=None):
    #     self.root = root
    #     self.transforms = transforms

    #     # Liste aller Sequenzen abrufen
    #     self.sequences = sorted(os.listdir(root))
    #     self.imgs = []
    #     self.masks = []

    #     print("Lade Sequenzen...")
    #     # Durch alle Sequenzen iterieren und Bilder/Masks sammeln
    #     for seq in tqdm(self.sequences, desc="Fortschritt", unit="seq"):
    #         seq_path = os.path.join(root, seq)
    #         img_file = os.path.join(seq_path, "step0.camera.png")
    #         mask_file = os.path.join(seq_path, "step0.camera.semantic segmentation.png")
    #         if os.path.exists(img_file) and os.path.exists(mask_file):
    #             self.imgs.append(img_file)
    #             self.masks.append(mask_file)

    # # def __getitem__(self, idx):
    # #     # Load image and mask
    # #     img_path = self.imgs[idx]
    # #     mask_path = self.masks[idx]
    # #     img = Image.open(img_path).convert("RGB")
    # #     mask = Image.open(mask_path).convert("RGBA")

    # #     # Convert mask to class IDs
    # #     mask_np = np.array(mask)
    # #     instance_mask = np.zeros(mask_np.shape[:2], dtype=np.int64)
    # #     for color, class_id in COLOR_TO_ID.items():
    # #         instance_mask[np.all(mask_np == np.array(color), axis=-1)] = class_id

    # #     # Get unique class IDs and generate binary masks
    # #     obj_ids = np.unique(instance_mask)
    # #     obj_ids = obj_ids[obj_ids > 0]  # Exclude background (0)
    # #     if len(obj_ids) == 0:
    # #         # Keine Objekte, setze ein leeres Bounding-Box und Labels
    # #         boxes = torch.zeros((0, 4), dtype=torch.float32)
    # #         labels = torch.zeros(0, dtype=torch.int64)
    # #         masks = torch.zeros((0, instance_mask.shape[0], instance_mask.shape[1]), dtype=torch.uint8)
    # #     else:
    # #         # Normale Verarbeitung
    # #         masks = instance_mask == obj_ids[:, None, None]
    # #         boxes = []
    # #         for i in range(len(obj_ids)):
    # #             pos = np.where(masks[i])
    # #             xmin = np.min(pos[1])
    # #             xmax = np.max(pos[1])
    # #             ymin = np.min(pos[0])
    # #             ymax = np.max(pos[0])
    # #             boxes.append([xmin, ymin, xmax, ymax])
    # #         boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # #         labels = torch.as_tensor(obj_ids, dtype=torch.int64)
    # #         masks = torch.as_tensor(masks, dtype=torch.uint8)

    # #     image_id = torch.tensor([idx])
    # #     if len(boxes) > 0:
    # #         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # #     else:
    # #         area = torch.tensor([0.0])  # Falls keine Boxen existieren, setze area auf 0
    # #     iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

    # #     target = {
    # #         "boxes": boxes,
    # #         "labels": labels,
    # #         "masks": masks,
    # #         "image_id": image_id,
    # #         "area": area,
    # #         "iscrowd": iscrowd,
    # #     }

    # #     if self.transforms is not None:
    # #         img = self.transforms(img)

    # #     return img, target
    # def __getitem__(self, idx):
    #     # Load image and mask
    #     img_path = self.imgs[idx]
    #     mask_path = self.masks[idx]
    #     img = Image.open(img_path).convert("RGB")
    #     mask = Image.open(mask_path).convert("RGBA")

    #     # Convert mask to class IDs
    #     mask_np = np.array(mask)
    #     instance_mask = np.zeros(mask_np.shape[:2], dtype=np.int64)
    #     for color, class_id in COLOR_TO_ID.items():
    #         instance_mask[np.all(mask_np == np.array(color), axis=-1)] = class_id

    #     # Get unique class IDs and generate binary masks
    #     obj_ids = np.unique(instance_mask)
    #     obj_ids = obj_ids[obj_ids > 0]  # Exclude background (0)

    #     if len(obj_ids) == 0:
    #         # No objects: set empty bounding boxes and labels
    #         boxes = torch.zeros((0, 4), dtype=torch.float32)
    #         labels = torch.zeros(0, dtype=torch.int64)
    #         masks = torch.zeros((0, instance_mask.shape[0], instance_mask.shape[1]), dtype=torch.uint8)
    #     else:
    #         # Normal processing
    #         masks = instance_mask == obj_ids[:, None, None]
    #         boxes = []
    #         for i in range(len(obj_ids)):
    #             pos = np.where(masks[i])
    #             xmin = np.min(pos[1])
    #             xmax = np.max(pos[1])
    #             ymin = np.min(pos[0])
    #             ymax = np.max(pos[0])
    #             boxes.append([xmin, ymin, xmax, ymax])
    #         boxes = torch.as_tensor(boxes, dtype=torch.float32)
    #         labels = torch.as_tensor(obj_ids, dtype=torch.int64)
    #         masks = torch.as_tensor(masks, dtype=torch.uint8)

    #     # Filter invalid bounding boxes
    #     if len(boxes) > 0:
    #         # Ensure positive height and width
    #         valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    #         boxes = boxes[valid_indices]
    #         labels = labels[valid_indices]
    #         masks = masks[valid_indices]

    #     image_id = torch.tensor([idx])
    #     if len(boxes) > 0:
    #         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    #     else:
    #         area = torch.tensor([0.0])  # Set area to 0 if no boxes exist
    #     iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

    #     target = {
    #         "boxes": boxes,
    #         "labels": labels,
    #         "masks": masks,
    #         "image_id": image_id,
    #         "area": area,
    #         "iscrowd": iscrowd,
    #     }

    #     if self.transforms is not None:
    #         img = self.transforms(img)

    #     return img, target

    def __len__(self):
        return len(self.imgs)
    
    def plot_samples(self, images, targets, num_samples=20):
        """
        Diese Funktion plottet die ersten 'num_samples' Bilder mit ihren
        zugehörigen Masken und den Labels der Klassen, wie sie für das Modell relevant sind.
        """
        print(num_samples) 
        for i in range(num_samples):
            image = images[i].numpy().transpose((1, 2, 0))  # Umwandeln von CxHxW in HxWxC
            target = targets[i]

            # Erstelle eine neue Figur für jedes Bild
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(image)
            ax.set_title(f"Sample {i+1}")

            # Plotte jede Maske einzeln
            for j in range(target['masks'].shape[0]):
                mask = target['masks'][j].numpy()

                # Nur den Maskenbereich auf das Bild anwenden
                masked_image = np.ma.masked_where(mask == 0, mask)  
                ax.imshow(masked_image, cmap='jet', alpha=0.5)  

                # Füge das Label der Klasse hinzu
                label = target['labels'][j].item()
                ax.text(
                    10, 10 + j * 20,  
                    f"Class: {label}",  
                    color='white', fontsize=12, backgroundcolor='black'
                )

            plt.show()

# Define transformations
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Load the pre-trained Mask R-CNN model
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

import time
from tqdm import tqdm

# Simple training loop
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    i = 0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}", ncols=100):
        # Move images and targets to the device (GPU)
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)

        # Total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimize
        losses.backward()
        optimizer.step()

        # Print the loss periodically
        if i % print_freq == 0:
            print(f"Epoch {epoch+1}, Iteration {i}, Loss: {losses.item()}")
        i += 1

def collate_fn(batch):
    return tuple(zip(*batch))

def calculate_iou(pred_mask, target_mask):
    intersection = (pred_mask & target_mask).sum()
    union = (pred_mask | target_mask).sum()
    iou = intersection / union if union > 0 else 0
    return iou

def evaluate_with_iou(model, data_loader, device):
    model.eval()
    total_iou = 0
    count = 0

    for images, targets in tqdm(data_loader, desc="Evaluating IoU", ncols=100):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)
        
        for output, target in zip(outputs, targets):
            pred_masks = output['masks'] > 0.5
            target_masks = target['masks']
            
            for pred_mask, target_mask in zip(pred_masks, target_masks):
                total_iou += calculate_iou(pred_mask.cpu().numpy(), target_mask.cpu().numpy())
                count += 1

    mean_iou = total_iou / count if count > 0 else 0
    print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou

if __name__ == "__main__":
    # Load the dataset
    root_dir = "C:/Users/Lukas/AppData/LocalLow/DefaultCompany/Fuwa_HDRP/solo_1/"  # Replace with your dataset directory
    dataset = BuildingSegmentationDataset(root=root_dir, transforms=get_transform(train=True))
    dataset_test = BuildingSegmentationDataset(root=root_dir, transforms=get_transform(train=False))

    # Split dataset into train and test
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])  # Last 50 for testing
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    # Load the pre-trained Mask R-CNN model
    num_classes = len(COLOR_TO_ID)  # Exclude the background class
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 10
    # for epoch in range(num_epochs):
    #     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    #     lr_scheduler.step()
    #     evaluate(model, data_loader_test, device)
    best_mean_iou = 0.0  # Initialisieren Sie den besten mAP-Wert

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Evaluieren Sie das Modell
        #mAP = evaluate(model, data_loader_test, device)
        #print(f"Epoch {epoch+1}/{num_epochs}, mAP: {mAP:.4f}")
        mean_iou = evaluate_with_iou(model, data_loader_test, device)
        print(f"Epoch {epoch+1}/{num_epochs}, mean_iou: {mean_iou:.4f}")

        # Speichern Sie das Modell der aktuellen Epoche
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        print(f"Modell für Epoche {epoch+1} gespeichert.")

        # Überprüfen Sie, ob dies das beste Modell ist
        if mean_iou > best_mean_iou:
            best_mAP = mean_iou
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Bestes Modell mit mAP {best_mAP:.4f} gespeichert.")
