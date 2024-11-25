import os
import torch
import torchvision
from references.engine import train_one_epoch, evaluate
import references.utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from tqdm import tqdm
from chatgpt_finetune_MaskRCNN import BuildingSegmentationDataset, get_transform, evaluate_with_iou, COLOR_TO_ID  # Annahme: Ihre Klasse ist in dieser Datei
from torch.utils.data import ConcatDataset

# Modellanpassung
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# Collate-Funktion für den DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

# # Hauptprogramm
# if __name__ == "__main__":
#     root_dir1 = "C:/Users/Lukas/AppData/LocalLow/DefaultCompany/Fuwa_HDRP/solo_25/"
#     root_dir2 = "C:\\Users\\Lukas\\AppData\\LocalLow\\DefaultCompany\\Fuwa_HDRP\\solo_2"
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     # Dataset laden
#     dataset1 = BuildingSegmentationDataset(root=root_dir1, transforms=get_transform(train=True))
#     dataset_test1 = BuildingSegmentationDataset(root=root_dir1, transforms=get_transform(train=False))

#     dataset2 = BuildingSegmentationDataset(root=root_dir2, transforms=get_transform(train=True))
#     dataset_test2 = BuildingSegmentationDataset(root=root_dir2, transforms=get_transform(train=False))

#     # Combine the datasets
#     dataset = ConcatDataset([dataset1, dataset2])

#     #imagess, targetss = zip(*[dataset[i] for i in range(5)])
#     #imagesss = list(imagess)  # Umwandeln in eine Liste
#     #targetsss = list(targetss)  # Umwandeln in eine Liste
#     #dataset.plot_samples(imagesss, targetsss, 5)
    
#     # Shuffle and limit to 5000 samples
#     torch.manual_seed(1)
#     indices = torch.randperm(len(dataset)).tolist()
#     train_data_limit = 100  # Limit to 10000 samples
#     test_data_limit = min(50, len(indices))
    
#     indices = indices[:train_data_limit]
#     train_indices = indices[:max(len(indices) - test_data_limit, 0)]
#     test_indices = indices[-test_data_limit:]
    
#     dataset = torch.utils.data.Subset(dataset, train_indices)

#     # Create test dataset (you can decide how to split test samples)
#     dataset_test = torch.utils.data.Subset(dataset,  test_indices)

#     # Create test dataset (you can decide how to split test samples)
#     dataset_test = torch.utils.data.Subset(dataset, indices[-50:])  # Use last 50 for testing
    
#     # # Datensatz aufteilen
#     # torch.manual_seed(1)
#     # indices = torch.randperm(len(dataset)).tolist()

#     # # Begrenzen der Datenanzahl auf 4000
#     # train_data_limit = 5000  # Anzahl der gewünschten Trainingsdaten
#     # dataset = torch.utils.data.Subset(dataset, indices[:train_data_limit])
#     # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
#     #dataset = torch.utils.data.Subset(dataset, indices[:-50])
#     #dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

#     # DataLoader erstellen
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
#     )
#     data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
#     )
    
#     # Modell erstellen
#     num_classes = len(COLOR_TO_ID)  # Anzahl der Klassen basierend auf COLOR_TO_ID
    
#     model = get_model_instance_segmentation(num_classes)
#     model.to(device)

#     # Optimierer und Scheduler definieren
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#     #evaluate(model, data_loader_test,device)

#     # Training und Evaluierung
#     num_epochs = 2
#     best_mean_iou = 0.0

#     for epoch in range(num_epochs):
#         train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
#         lr_scheduler.step()

#         # Evaluierung basierend auf IoU
#         mean_iou = evaluate_with_iou(model, data_loader_test, device)
#         print(f"Epoch {epoch+1}/{num_epochs}, mean IoU: {mean_iou:.4f}")

#         # Modell speichern
#         torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
#         print(f"Modell für Epoche {epoch+1} gespeichert.")

#         #evaluate(model, data_loader_test,device)

#         # Bestes Modell speichern
#         if mean_iou > best_mean_iou:
#             best_mean_iou = mean_iou
#             torch.save(model.state_dict(), "best_model.pth")
#             print(f"Bestes Modell mit mean IoU {best_mean_iou:.4f} gespeichert.")


# Hauptprogramm
if __name__ == "__main__":
    # Wurzeldirektoren für die Datensätze
    #root_dir1 = "C:/Users/Lukas/AppData/LocalLow/DefaultCompany/Fuwa_HDRP/solo_25/"
    root_dir2 = "C:\\Users\\Lukas\\AppData\\LocalLow\\DefaultCompany\\Fuwa_HDRP\\solo_2"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Datensätze laden
    #dataset1 = BuildingSegmentationDataset(root=root_dir1, transforms=get_transform(train=True))
    dataset2 = BuildingSegmentationDataset(root=root_dir2, transforms=get_transform(train=True))

    # Datensätze zusammenführen
    #combined_dataset = ConcatDataset([dataset1, dataset2])

    # Begrenzen der Datenanzahl
    torch.manual_seed(1)
    total_data_limit = 15000  # Anzahl der Trainingsdaten begrenzen
    test_data_limit = 50  # Anzahl der Testdaten begrenzen

    # Sicherstellen, dass wir nicht mehr Indizes anfordern als verfügbar
    #available_indices = min(len(combined_dataset), total_data_limit + test_data_limit)
    available_indices = min(len(dataset2), total_data_limit + test_data_limit)
    indices = torch.randperm(available_indices).tolist()

    # Aufteilen in Trainings- und Testdaten
    train_indices = indices[:total_data_limit]
    test_indices = indices[total_data_limit:total_data_limit + test_data_limit]

    # Subsets erstellen
    #train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    #test_dataset = torch.utils.data.Subset(combined_dataset, test_indices)
    train_dataset = torch.utils.data.Subset(dataset2, train_indices)
    test_dataset = torch.utils.data.Subset(dataset2, test_indices)

    # DataLoader erstellen
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Modell erstellen
    num_classes = len(COLOR_TO_ID)  # Anzahl der Klassen basierend auf COLOR_TO_ID
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Optimierer und Scheduler definieren
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training und Evaluierung
    num_epochs = 2
    best_mean_iou = 0.0

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Evaluierung basierend auf IoU
        mean_iou = evaluate_with_iou(model, data_loader_test, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, mean IoU: {mean_iou:.4f}")

        # Modell speichern
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        print(f"Modell für Epoche {epoch + 1} gespeichert.")

        # Bestes Modell speichern
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Bestes Modell mit mean IoU {best_mean_iou:.4f} gespeichert.")
