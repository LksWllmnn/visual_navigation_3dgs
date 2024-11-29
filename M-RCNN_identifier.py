# import matplotlib.pyplot as plt
# from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
# from PIL import Image
# import torch
# from chatgpt_finetune_MaskRCNN import get_model_instance_segmentation, get_transform
# from torchvision.transforms import v2 as T

# # IDs zu Gebäudenamen zuordnen
# ID_TO_NAME = {
#     0: "Background",
#     1: "A-Building",
#     2: "B-Building",
#     3: "C-Building",
#     4: "E-Building",
#     5: "F-Building",
#     6: "G-Building",
#     7: "H-Building",
#     8: "I-Building",
#     9: "L-Building",
#     10: "M-Building",
#     11: "N-Building",
#     12: "O-Building",
#     13: "R-Building",
#     14: "Z-Building",
#     #15: "Other"
# }

# # Modell auf Gerät laden
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # Modell initialisieren
# num_classes = len(ID_TO_NAME)
# model = get_model_instance_segmentation(num_classes)
# model.load_state_dict(torch.load("model_epoch_2.pth", map_location=device))
# model.to(device)
# model.eval()

# # Transformation für Evaluation laden
# eval_transform = get_transform(train=False)

# # Bild laden und vorbereiten
# image_path = "D:\\Thesis\\data\\gsplat_vid_images\\00040.jpg"
# image = Image.open(image_path).convert("RGB")
# image_tensor = eval_transform(image).to(device).unsqueeze(0)

# # Vorhersagen berechnen
# with torch.no_grad():
#     predictions = model(image_tensor)
#     pred = predictions[0]

# # Originalbild für Darstellung
# image_display = T.ToTensor()(image).mul(255).byte()

# # Masken, Bounding Boxes, Labels und Scores vorbereiten
# confidence_threshold = 0.5  # Setze die Confidence-Schwelle
# labels = pred["labels"]
# scores = pred["scores"]
# boxes = pred["boxes"]
# masks = (pred["masks"] > confidence_threshold).squeeze(1)

# # Indizes der Bounding Boxes filtern basierend auf Confidence-Wahrscheinlichkeit
# valid_indices = [
#     i for i, (label, score) in enumerate(zip(labels, scores)) 
#     if label.item() != 15 and score >= confidence_threshold
# ]
# filtered_boxes = boxes[valid_indices]
# filtered_labels = labels[valid_indices]
# filtered_scores = scores[valid_indices]
# filtered_masks = masks[valid_indices]

# class_colors = ["blue", "green", "yellow", "pink", "cyan"]
# mask_colors = [class_colors[label.item() % len(class_colors)] for label in filtered_labels]

# # Gebäudenamen als Labels vorbereiten
# pred_labels = [
#     f"{ID_TO_NAME[label.item()]}: {score:.3f}"
#     for label, score in zip(filtered_labels, filtered_scores)
# ]

# output_image = draw_segmentation_masks(image_display, filtered_masks, alpha=0.5, colors=mask_colors)

# # Bounding Boxes zeichnen
# output_image = draw_bounding_boxes(
#     output_image, 
#     filtered_boxes.long(), 
#     labels=pred_labels, 
#     colors="red"
# )

# # Ergebnis anzeigen
# plt.figure(figsize=(12, 12))
# plt.imshow(output_image.permute(1, 2, 0))
# plt.axis("off")
# plt.show()



# import os
# import matplotlib.pyplot as plt
# from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
# from PIL import Image
# import torch
# from chatgpt_finetune_MaskRCNN import get_model_instance_segmentation, get_transform
# from torchvision.transforms import v2 as T

# # IDs zu Gebäudenamen zuordnen
# ID_TO_NAME = {
#     0: "Background",
#     1: "A-Building",
#     2: "B-Building",
#     3: "C-Building",
#     4: "E-Building",
#     5: "F-Building",
#     6: "G-Building",
#     7: "H-Building",
#     8: "I-Building",
#     9: "L-Building",
#     10: "M-Building",
#     11: "N-Building",
#     12: "O-Building",
#     13: "R-Building",
#     14: "Z-Building",
#     # 15: "Other"
# }

# # Modell auf Gerät laden
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # Modell initialisieren
# num_classes = len(ID_TO_NAME)
# model = get_model_instance_segmentation(num_classes)
# model.load_state_dict(torch.load("chkpts\\best_m-RCNN_model_6_inScene15000.pth", map_location=device))
# model.to(device)
# model.eval()

# # Transformation für Evaluation laden
# eval_transform = get_transform(train=False)

# def analyze_and_save_images(input_folder, output_folder, confidence_threshold=0.5):
#     """
#     Analysiert alle Bilder in einem Ordner und speichert die Ergebnisse im angegebenen Ausgabeverzeichnis.
#     """
#     # Erstelle den Ausgabeordner, falls er nicht existiert
#     os.makedirs(output_folder, exist_ok=True)

#     # Liste aller Bilder im Eingabeordner
#     image_files = [
#         f for f in os.listdir(input_folder) 
#         if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#     ]

#     # Analysiere jedes Bild
#     for image_file in image_files:
#         image_path = os.path.join(input_folder, image_file)
#         output_path = os.path.join(output_folder, image_file)

#         # Bild laden und vorbereiten
#         image = Image.open(image_path).convert("RGB")
#         image_tensor = eval_transform(image).to(device).unsqueeze(0)

#         # Vorhersagen berechnen
#         with torch.no_grad():
#             predictions = model(image_tensor)
#             pred = predictions[0]

#         # Originalbild für Darstellung
#         image_display = T.ToTensor()(image).mul(255).byte()

#         # Masken, Bounding Boxes, Labels und Scores vorbereiten
#         labels = pred["labels"]
#         scores = pred["scores"]
#         boxes = pred["boxes"]
#         masks = (pred["masks"] > confidence_threshold).squeeze(1)

#         # Indizes der Bounding Boxes filtern basierend auf Confidence-Wahrscheinlichkeit
#         valid_indices = [
#             i for i, (label, score) in enumerate(zip(labels, scores)) 
#             if label.item() != 15 and score >= confidence_threshold
#         ]
#         filtered_boxes = boxes[valid_indices]
#         filtered_labels = labels[valid_indices]
#         filtered_scores = scores[valid_indices]
#         filtered_masks = masks[valid_indices]

#         class_colors = ["blue", "green", "yellow", "pink", "cyan"]
#         mask_colors = [class_colors[label.item() % len(class_colors)] for label in filtered_labels]

#         # Gebäudenamen als Labels vorbereiten
#         pred_labels = [
#             f"{ID_TO_NAME[label.item()]}: {score:.3f}"
#             for label, score in zip(filtered_labels, filtered_scores)
#         ]

#         # Masken und Bounding Boxes auf das Bild zeichnen
#         output_image = draw_segmentation_masks(image_display, filtered_masks, alpha=0.5, colors=mask_colors)
#         output_image = draw_bounding_boxes(output_image, filtered_boxes.long(), labels=pred_labels, colors="red")

#         # Speichern des Ergebnisses
#         output_image = output_image.permute(1, 2, 0).cpu().numpy()
#         plt.imsave(output_path, output_image)
#         print(f"Saved analyzed image to {output_path}")

# # Beispielverwendung
# input_folder = "D:\\Thesis\\data\\gsplat_vid_images"  # Pfad zum Eingabeordner
# output_folder = "D:\\Thesis\\data\\analyzed_images_M-RCNN"  # Pfad zum Ausgabeverzeichnis
# analyze_and_save_images(input_folder, output_folder)


import os
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from PIL import Image
import torch
from chatgpt_finetune_MaskRCNN import get_model_instance_segmentation, get_transform
from torchvision.transforms import v2 as T

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

# Modell auf Gerät laden
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Modell initialisieren
num_classes = len(ID_TO_NAME)
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("chkpts\\best_m-RCNN_model_6_inScene15000.pth", map_location=device))
model.to(device)
model.eval()

# Transformation für Evaluation laden
eval_transform = get_transform(train=False)

def analyze_and_save_images(input_folder, output_folder, confidence_threshold=0.5):
    """
    Analysiert alle Bilder in einem Ordner und speichert die Ergebnisse im angegebenen Ausgabeverzeichnis.
    """
    # Erstelle den Ausgabeordner, falls er nicht existiert
    os.makedirs(output_folder, exist_ok=True)

    # Liste aller Bilder im Eingabeordner
    image_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Analysiere jedes Bild
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Bild laden und vorbereiten
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

        # Indizes der Bounding Boxes filtern basierend auf Confidence-Wahrscheinlichkeit
        valid_indices = [
            i for i, (label, score) in enumerate(zip(labels, scores)) 
            if label.item() != 15 and score >= confidence_threshold
        ]
        filtered_boxes = boxes[valid_indices]
        filtered_labels = labels[valid_indices]
        filtered_scores = scores[valid_indices]
        filtered_masks = masks[valid_indices]

        # Spezifische Farben für Masken basierend auf Gebäudenamen
        mask_colors = [ID_TO_COLOR[label.item()] for label in filtered_labels]

        # Gebäudenamen als Labels vorbereiten
        pred_labels = [
            f"{ID_TO_NAME[label.item()]}: {score:.3f}"
            for label, score in zip(filtered_labels, filtered_scores)
        ]

        # Masken und Bounding Boxes auf das Bild zeichnen
        output_image = draw_segmentation_masks(image_display, filtered_masks, alpha=0.5, colors=mask_colors)
        output_image = draw_bounding_boxes(output_image, filtered_boxes.long(), labels=pred_labels, colors="red")

        # Speichern des Ergebnisses
        output_image = output_image.permute(1, 2, 0).cpu().numpy()
        plt.imsave(output_path, output_image)
        print(f"Saved analyzed image to {output_path}")

# Beispielverwendung
input_folder = "D:\\Thesis\\data\\gsplat_vid_images"  # Pfad zum Eingabeordner
output_folder = "D:\\Thesis\\data\\M-RCNN_analyzed_2"  # Pfad zum Ausgabeverzeichnis
analyze_and_save_images(input_folder, output_folder)

