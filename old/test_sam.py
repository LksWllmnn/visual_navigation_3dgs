import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def save_masked_image(image, mask, output_path, mask_index):
    # Maske anwenden: Nur den Bereich behalten, wo die Maske aktiv ist
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

    # Neues Bild speichern
    output_file = os.path.join(output_path, f"masked_image_{mask_index}.png")
    cv2.imwrite(output_file, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

def process_image(image_path, mask_generator, output_path):
    # Bild einlesen
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load image {image_path}, skipping...")
        return
    
    # Bild in RGB umwandeln
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Timer starten
    start_time = time.time()
    
    # Maske generieren
    masks = mask_generator.generate(image)

    # Timer stoppen
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to generate masks: {elapsed_time:.2f} seconds")
    
    # Anzahl der Masken und Schlüssel drucken (optional)
    print(f"Image: {image_path}, Masks generated: {len(masks)}")
    
    # Speichern der Maskenbereiche
    for idx, mask in enumerate(masks):
        segmentation_mask = mask['segmentation']
        save_masked_image(image, segmentation_mask, output_path, idx)
    
    # Bild anzeigen
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

# SAM initialisieren
device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="F:\\Studium\\Master\\Thesis\\extRepos\\sam\\segment-anything\\chkpts\\sam_vit_h_4b8939.pth")
sam.to(device=device)

# Maskengenerator initialisieren
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

# Pfad zum Ordner mit den Bildern
image_folder = "./data/"
output_folder = "./output_masks/"

# Ausgabeordner erstellen, falls nicht vorhanden
os.makedirs(output_folder, exist_ok=True)

# Alle Bilddateien im Ordner auflisten
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# Jedes Bild verarbeiten
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    process_image(image_path, mask_generator, output_folder)
    break