import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from references.engine import train_one_epoch  # Importiere deine Trainingsfunktion
from dataset_class import ImageTitleDatasetMRCNN, BUILDING_COLORS
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision

# Collate-Funktion für den DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

# Load the pre-trained Mask R-CNN model
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# Berechnung von IoU
def calculate_iou(pred_masks, true_masks):
    """
    Berechnet den IoU zwischen vorhergesagten und echten Masken.
    :param pred_masks: Vorhergesagte Masken (Numpy-Array oder Tensor)
    :param true_masks: Echte Masken (Numpy-Array oder Tensor)
    :return: Durchschnittliche IoU
    """
    intersection = (pred_masks & true_masks).sum(dim=(1, 2))  # True Positive
    union = (pred_masks | true_masks).sum(dim=(1, 2))  # Union: TP + FP + FN
    iou = intersection / (union + 1e-6)  # Vermeidung von Division durch 0
    return iou.mean().item()  # Mittelwert über alle Masken

# Berechnung der Pixel-Level-Accuracy
def calculate_segmentation_accuracy(model, data_loader, device):
    """
    Berechnet die Pixel-Level-Accuracy für das Segmentierungsmodell.
    :param model: Das Mask R-CNN-Modell
    :param data_loader: DataLoader für Validierungsdaten
    :param device: Torch-Device (CPU/GPU)
    :return: Durchschnittliche Accuracy
    """
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                true_mask = target['masks'].cpu().numpy() > 0.5
                pred_mask = output['masks'].cpu().detach().numpy() > 0.5

                correct_pixels += (pred_mask == true_mask).sum()
                total_pixels += true_mask.size

    return correct_pixels / total_pixels

# Training und Validierung
def train_and_evaluate(model, train_loader, val_loader, device, num_epochs, lr, save_path="best_model.pth"):
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    train_losses, val_losses, iou_scores, accuracies = [], [], [], []
    best_iou = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss, mean_iou, accuracy = 0, 0, 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)
                for target, output in zip(targets, outputs):
                    true_mask = target['masks'] > 0.5
                    pred_mask = output['masks'] > 0.5

                    val_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                        pred_mask.float(), true_mask.float()
                    ).item()
                    mean_iou += calculate_iou(pred_mask, true_mask)
                    accuracy += calculate_segmentation_accuracy(model, val_loader, device)

        val_losses.append(val_loss / len(val_loader))
        iou_scores.append(mean_iou / len(val_loader))
        accuracies.append(accuracy / len(val_loader))

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_losses[-1]:.4f}, IoU={iou_scores[-1]:.4f}, Accuracy={accuracies[-1]:.4f}")

        # Update Best Model
        if iou_scores[-1] > best_iou:
            best_iou = iou_scores[-1]
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with IoU: {best_iou:.4f}")

        lr_scheduler.step()

    return train_losses, val_losses, iou_scores, accuracies

# Plotting-Funktion
def save_plots_and_logs(train_losses, val_losses, iou_scores, accuracies, log_path="training_log.txt", plot_dir="plots"):
    import os
    os.makedirs(plot_dir, exist_ok=True)

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.savefig(f"{plot_dir}/loss_plot.png")
    plt.close()

    # IoU Plot
    plt.figure(figsize=(10, 5))
    plt.plot(iou_scores, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU per Epoch")
    plt.legend()
    plt.savefig(f"{plot_dir}/iou_plot.png")
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.savefig(f"{plot_dir}/accuracy_plot.png")
    plt.close()

    # Logs speichern
    with open(log_path, "w") as log_file:
        log_file.write("Epoch\tTrain Loss\tValidation Loss\tIoU\tAccuracy\n")
        for epoch in range(len(train_losses)):
            log_file.write(f"{epoch+1}\t{train_losses[epoch]:.4f}\t{val_losses[epoch]:.4f}\t{iou_scores[epoch]:.4f}\t{accuracies[epoch]:.4f}\n")
    print(f"Logs und Plots gespeichert in {plot_dir}")

# Hauptprogramm
if __name__ == "__main__":
    # Daten laden
    root_dir = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\building_surround_pictures"
    #root_dir = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\scene_building_pictures"
    #root_dir = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\building_big_surround_pictures"
    
    dataset = ImageTitleDatasetMRCNN(root_dir=root_dir, filter=False)
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))  # 80% für Training
    val_size = len(dataset) - train_size  # 20% für Validierung/Test

    # Zufällige Aufteilung des Datasets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Modell initialisieren
    num_classes = len(BUILDING_COLORS)  # Anzahl der Klassen ohne Hintergrund
    model = get_model_instance_segmentation(num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Training
    train_losses, val_losses, iou_scores, accuracies = train_and_evaluate(
        model, train_loader, val_loader, device, num_epochs=2, lr=0.005
    )

    # Plots und Logs speichern
    save_plots_and_logs(train_losses, val_losses, iou_scores, accuracies)
