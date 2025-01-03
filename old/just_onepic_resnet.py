import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

def save_anns(anns, image, output_path, mask_color=[255, 0, 0], transparency=90):
    if len(anns) == 0:
        return
    
    # Kopie des Originalbilds für Maskenüberlagerung
    image_with_mask = image.copy()
    
    for ann in anns:
        m = ann['segmentation']
        
        # Erstelle eine Maske mit fester Farbe und Transparenz
        overlay = np.zeros((*m.shape, 4), dtype=np.uint8)
        overlay[m] = np.concatenate([mask_color, [transparency]])

        # Überlagere die Maske mit Alpha-Blending
        alpha_mask = overlay[:, :, 3] / 255.0
        for c in range(3):  # RGB-Kanäle
            image_with_mask[:, :, c] = (1 - alpha_mask) * image_with_mask[:, :, c] + alpha_mask * overlay[:, :, c]

    # Speichern des Bilds in voller Auflösung
    image_rgb = cv2.cvtColor(image_with_mask, cv2.COLOR_RGBA2RGB)
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

def classify_images(model, label_mapping, images, target_class, confidence_threshold=50):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    images_tensor = torch.stack([preprocess(Image.fromarray(img)).to(device) for img in images])

    with torch.no_grad():
        outputs = model(images_tensor)
        probs = F.softmax(outputs, dim=1)
        
    valid_images, valid_masks = [], []
    target_class_idx = None
    
    # Finde den Index der Zielklasse im Label-Mapping
    for idx, class_name in label_mapping.items():
        if class_name == target_class:
            target_class_idx = idx
            break
    
    if target_class_idx is None:
        raise ValueError(f"Target class '{target_class}' not found in label mapping.")

    # Überprüfe jede Maske, ob die Wahrscheinlichkeit der Zielklasse den Schwellwert überschreitet
    for idx, prob in enumerate(probs):
        target_prob = prob[target_class_idx].item() * 100
        if target_prob >= confidence_threshold:
            valid_images.append(images[idx])
            valid_masks.append(idx)
    
    return valid_images, valid_masks

def process_single_image(image_path, output_folder, mask_generator, model, label_mapping, target_class, confidence_threshold=20):
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Masken generieren
    masks = mask_generator.generate(image)
    
    # Ordner für Masken und maskierte Bilder
    masks_output_path = os.path.join(output_folder, "masks")
    mask_images_output_path = os.path.join(output_folder, "mask_images")
    os.makedirs(masks_output_path, exist_ok=True)
    os.makedirs(mask_images_output_path, exist_ok=True)
    
    mask_images = []
    for idx, mask in enumerate(masks):
        # Speichere Maske als PNG
        mask_output_file = os.path.join(masks_output_path, f"mask_{idx}.png")
        mask_array = (mask['segmentation'] * 255).astype(np.uint8)
        cv2.imwrite(mask_output_file, mask_array)
        
        # Erstelle und speichere maskiertes Bild
        mask_image = cv2.bitwise_and(image, image, mask=mask['segmentation'].astype(np.uint8))
        mask_images.append(mask_image)
        mask_image_output_file = os.path.join(mask_images_output_path, f"mask_image_{idx}.jpg")
        cv2.imwrite(mask_image_output_file, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
    
    # Gültige Masken basierend auf Klassifikation finden
    valid_images, valid_mask_indices = classify_images(
        model=model, label_mapping=label_mapping, images=mask_images, target_class=target_class, confidence_threshold=confidence_threshold
    )
    
    # Bild mit gültigen Masken speichern
    output_image_path = os.path.join(output_folder, "output_image_with_masks.jpg")
    if valid_images:
        valid_masks = [masks[i] for i in valid_mask_indices]
        save_anns(valid_masks, image, output_image_path)
        print(f"Saved marked image to {output_image_path}")
    else:
        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"No valid masks for class '{target_class}' in image: {image_path}")
        
# Initialisierung des Modells und Maskengenerators
device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="chkpts\\sam_vit_h_4b8939.pth")
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam, points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92, crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100
)

model = models.resnet50(pretrained=False)
num_classes = 15
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("chkpts\\resnet50_finetuned_1.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

label_mapping = {
    0: "A-Building", 1: "B-Building", 2: "C-Building", 3: "E-Building", 4: "F-Building", 5: "G-Building",
    6: "H-Building", 7: "I-Building", 8: "L-Building", 9: "M-Building", 10: "N-Building", 11: "O-Building",
    12: "R-Building", 13: "Z-Building", 14: "other"
}

if __name__ == "__main__":
    image_path = "D:\\Thesis\\data\\gsplat_vid_images\\00002.jpg"
    output_folder = "D:\\Thesis\\data\\resnet_gsplat_testFolder\\example_output"
    target_class = "B-Building"
    process_single_image(image_path, output_folder, mask_generator, model, label_mapping, target_class)
