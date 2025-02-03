# based on https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 14.01.2025

import os
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time

def measure_time(func):
    """Dekorator zur Messung der Ausführungszeit einer Funktion."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} dauerte {end_time - start_time:.4f} Sekunden.")
        return result
    return wrapper

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

# IDs zu Gebäudenamen zuordnen
ID_TO_NAME = {
    0: "Background",
    1: "A-Building",
    2: "B-Building",
    3: "C-Building",
    4: "E-Building",
    5: "F-Building",
    6: "G-Building",
    7: "H-Building",
    8: "I-Building",
    9: "L-Building",
    10: "M-Building",
    11: "N-Building",
    12: "O-Building",
    13: "R-Building",
    14: "Z-Building",
}

# Farben für Gebäude zuordnen
ID_TO_COLOR = {
    0: "gray",       # Background
    1: "blue",       # A-Building
    2: "green",      # B-Building
    3: "yellow",     # C-Building
    4: "orange",     # E-Building
    5: "cyan",       # F-Building
    6: "purple",     # G-Building
    7: "pink",       # H-Building
    8: "red",        # I-Building
    9: "brown",      # L-Building
    10: "magenta",   # M-Building
    11: "lime",      # N-Building
    12: "turquoise", # O-Building
    13: "gold",      # R-Building
    14: "navy",      # Z-Building
}

@measure_time
def analyze_and_save_images(input_folder, output_folder, model, device, eval_transform, confidence_threshold=0.5,  ):
    """
    Analysiert alle Bilder in einem Ordner und speichert die Ergebnisse in den angegebenen Ausgabeverzeichnissen.
    """
    # Ordnerstruktur erstellen
    combined_folder = os.path.join(output_folder, "combined")
    qualitative_folder = os.path.join(output_folder, "qualitative")
    just_mask_folder = os.path.join(output_folder, "just-mask")
    os.makedirs(combined_folder, exist_ok=True)
    os.makedirs(qualitative_folder, exist_ok=True)
    os.makedirs(just_mask_folder, exist_ok=True)
    for class_name in ID_TO_NAME.values():
        if class_name != "Background":
            os.makedirs(os.path.join(qualitative_folder, class_name), exist_ok=True)
            os.makedirs(os.path.join(just_mask_folder, class_name), exist_ok=True)

    # Liste aller Bilder im Eingabeordner
    image_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Analysiere jedes Bild
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = eval_transform(image).to(device).unsqueeze(0)

        # Vorhersagen berechnen
        with torch.no_grad():
            predictions = model(image_tensor)
            pred = predictions[0]

        # Originalbild für Darstellung
        image_display = T.ToTensor()(image).mul(255).byte()

        # Masken, Bounding Boxes, Labels und Scores vorbereiten
        labels = pred["labels"]
        scores = pred["scores"]
        boxes = pred["boxes"]
        masks = (pred["masks"] > confidence_threshold).squeeze(1)

        # Spezifische Farben für Masken basierend auf Gebäudenamen
        mask_colors = [ID_TO_COLOR[label.item()] for label in labels]

        # COMBINED OUTPUT
        valid_indices = [
            i for i, score in enumerate(scores) if score >= confidence_threshold
        ]
        filtered_boxes = boxes[valid_indices]
        filtered_labels = labels[valid_indices]
        filtered_masks = masks[valid_indices]

        pred_labels = [
            f"{ID_TO_NAME[label.item()]}: {score:.3f}"
            for label, score in zip(filtered_labels, scores[valid_indices])
        ]

        combined_image = draw_segmentation_masks(image_display, filtered_masks, alpha=0.5, colors=mask_colors)
        combined_image = draw_bounding_boxes(combined_image, filtered_boxes.long(), labels=pred_labels, colors="red")
        combined_output_path = os.path.join(combined_folder, image_file)
        combined_image = combined_image.permute(1, 2, 0).cpu().numpy()
        plt.imsave(combined_output_path, combined_image)

        # QUALITATIVE AND JUST-MASK OUTPUTS
        for class_id, class_name in ID_TO_NAME.items():
            if class_name == "Background":
                continue

            # Filter für die aktuelle Klasse
            class_indices = [i for i, label in enumerate(labels) if label.item() == class_id and scores[i] >= confidence_threshold]
            class_masks = masks[class_indices]
            

            # Qualitative Output: RGB + Maske
            qualitative_image = draw_segmentation_masks(image_display.clone(), class_masks, alpha=0.5, colors=ID_TO_COLOR[class_id])
            qualitative_output_path = os.path.join(qualitative_folder, class_name, image_file)
            qualitative_image = qualitative_image.permute(1, 2, 0).cpu().numpy()
            plt.imsave(qualitative_output_path, qualitative_image)

            # Just-Mask Output: Schwarzer Hintergrund + Maske
            black_background = torch.zeros_like(image_display)
            just_mask_image = draw_segmentation_masks(black_background, class_masks, alpha=1.0, colors=ID_TO_COLOR[class_id])
            just_mask_output_path = os.path.join(just_mask_folder, class_name, image_file)
            just_mask_image = just_mask_image.permute(1, 2, 0).cpu().numpy()
            plt.imsave(just_mask_output_path, just_mask_image)

        print(f"Processed and saved outputs for {image_file}")

# Beispielverwendung
# input_folder = r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\output\test-pics-controll"  # Pfad zum Eingabeordner

#output_folder = r"F:\Studium\Master\Thesis\data\final_final_results\scene_mrcnn"  # Pfad zum Ausgabeverzeichnis
#output_folder= r"F:\Studium\Master\Thesis\data\final_final_results\surround_mrcnn"
#output_folder= r"F:\Studium\Master\Thesis\data\final_final_results\big-surround_mrcnn"


def init(model_path, case_name, input_folder):
    # Modell auf Gerät laden
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Modell initialisieren
    num_classes = len(ID_TO_NAME)
    model = get_model_instance_segmentation(num_classes)
    if case_name != "no-finetuning":
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("... no finetuning")
    #model.load_state_dict(torch.load(r"F:\Studium\Master\Thesis\chkpts\MRCNN-Models\scene_mrcnn_model.pth", map_location=device))
    #model.load_state_dict(torch.load(r"F:\Studium\Master\Thesis\chkpts\MRCNN-Models\surround_mrcnn_model.pth", map_location=device))
    #model.load_state_dict(torch.load(r"F:\Studium\Master\Thesis\chkpts\MRCNN-Models\big-surround_mrcnn_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Transformation für Evaluation laden
    eval_transform = get_transform(train=False)
    output_folder= f"F:\\Studium\\Master\\Thesis\\data\\timetest\\mrcnn_{case_name}"
    analyze_and_save_images(input_folder, output_folder, model=model, device=device, eval_transform=eval_transform)

input_folder = r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\feature-splatting\rgb"

#no-finetuning
model_path = r""
init(model_path=model_path, case_name="no-finetuning", input_folder=input_folder)

#Big-surround
model_path = r"F:\Studium\Master\Thesis\chkpts\MRCNN-Models\big-surround_mrcnn_model.pth"
init(model_path=model_path, case_name="big-surround", input_folder=input_folder)

#scene
model_path = r"F:\Studium\Master\Thesis\chkpts\MRCNN-Models\scene_mrcnn_model.pth"
init(model_path=model_path, case_name="scene", input_folder=input_folder)

#scene
model_path = r"F:\Studium\Master\Thesis\chkpts\MRCNN-Models\surround_mrcnn_model.pth"
init(model_path=model_path, case_name="surround", input_folder=input_folder)