# import os
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
# import torch
# from torchvision import transforms, models
# import torch.nn.functional as F
# from PIL import Image
# import argparse

# def show_anns(anns, image):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     plt.imshow(image)
#     ax.imshow(img)

# def classify_images(model, label_mapping, images, target_class, confidence_threshold=50):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
    
#     images_pil = [Image.fromarray(img) for img in images]
#     images_tensor = torch.stack([preprocess(img).to(device) for img in images_pil])

#     with torch.no_grad():
#         outputs = model(images_tensor)
#         probs = F.softmax(outputs, dim=1)
#         top_probs, top_labels = probs.topk(1, dim=1)

#     valid_images = []
#     valid_masks = []

#     for idx, (top_label, top_prob) in enumerate(zip(top_labels, top_probs)):
#         confidence = top_prob.item() * 100
#         predicted_class = label_mapping[top_label.item()]

#         if confidence >= confidence_threshold and predicted_class == target_class:
#             valid_images.append(images[idx])
#             valid_masks.append(idx)

#     return valid_images, valid_masks

# def process_image(image_path, mask_generator, model, label_mapping, target_class, confidence_threshold=50):
#     image = cv2.imread(image_path)
    
#     if image is None:
#         print(f"Could not load image {image_path}, skipping...")
#         return
    
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     start_time = time.time()
    
#     masks = mask_generator.generate(image)

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Time to generate masks: {elapsed_time:.2f} seconds")
    
#     print(f"Image: {image_path}, Masks generated: {len(masks)}")

#     mask_images = []
#     for mask in masks:
#         segmentation_mask = mask['segmentation']
#         masked_image = cv2.bitwise_and(image, image, mask=segmentation_mask.astype(np.uint8))
#         mask_images.append(masked_image)

#     valid_images, valid_mask_indices = classify_images(
#         model=model,
#         label_mapping=label_mapping,
#         images=mask_images,
#         target_class=target_class,
#         confidence_threshold=confidence_threshold
#     )
    
#     if len(valid_images) == 0:
#         print(f"No valid masks for class '{target_class}' in image: {image_path}")
#         return
    
#     plt.figure(figsize=(20, 20))
#     plt.imshow(image)
#     valid_masks = [masks[i] for i in valid_mask_indices]
#     show_anns(valid_masks, image)
#     plt.axis('off')
#     plt.show()

# device = "cuda"
# sam = sam_model_registry["vit_h"](checkpoint="F:\\Studium\\Master\\Thesis\\extRepos\\sam\\segment-anything\\chkpts\\sam_vit_h_4b8939.pth")
# sam.to(device=device)

# # Maskengenerator initialisieren
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=4,
#     pred_iou_thresh=0.86,
#     stability_score_thresh=0.92,
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,
# )

# model = models.mobilenet_v2(pretrained=False)
# num_classes = 15
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
# model.load_state_dict(torch.load("./chkpt/mobilenetv2_finetuned_4.pth", map_location=device, weights_only=True))
# model.to(device)
# model.eval()

# label_mapping = {
#     0: "A-Building",
#     1: "B-Building",
#     2: "C-Building",
#     3: "E-Building",
#     4: "F-Building",
#     5: "G-Building",
#     6: "H-Building",
#     7: "I-Building",
#     8: "L-Building",
#     9: "M-Building",
#     10: "N-Building",
#     11: "O-Building",
#     12: "R-Building",
#     13: "Z-Building",
#     14: "other"
# }

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process images and filter masks by building class.")
    
#     parser.add_argument(
#         "--building_class", 
#         type=str, 
#         required=True,
#         help="The name of the building class to keep (e.g., 'A-Building')"
#     )

#     args = parser.parse_args()

#     target_class = args.building_class
#     if target_class not in label_mapping.values():
#         print(f"Invalid building class: {target_class}")
#         exit(1)

#     image_folder = "./data/"

#     image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

#     for image_file in image_files:
#         image_path = os.path.join(image_folder, image_file)
#         process_image(image_path, mask_generator, model, label_mapping, target_class)
#         break

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import argparse

def show_anns(anns, image):
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
    plt.imshow(image)
    ax.imshow(img)

def classify_images(model, label_mapping, images, target_class, confidence_threshold=50):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    images_pil = [Image.fromarray(img) for img in images]
    images_tensor = torch.stack([preprocess(img).to(device) for img in images_pil])

    with torch.no_grad():
        outputs = model(images_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_labels = probs.topk(1, dim=1)

    valid_images = []
    valid_masks = []

    for idx, (top_label, top_prob) in enumerate(zip(top_labels, top_probs)):
        confidence = top_prob.item() * 100
        predicted_class = label_mapping[top_label.item()]

        if confidence >= confidence_threshold and predicted_class == target_class:
            valid_images.append(images[idx])
            valid_masks.append(idx)

    return valid_images, valid_masks

def process_image(image_path, mask_generator, model, label_mapping, target_class, confidence_threshold=50):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load image {image_path}, skipping...")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    
    masks = mask_generator.generate(image)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to generate masks: {elapsed_time:.2f} seconds")
    
    print(f"Image: {image_path}, Masks generated: {len(masks)}")

    mask_images = []
    for mask in masks:
        segmentation_mask = mask['segmentation']
        masked_image = cv2.bitwise_and(image, image, mask=segmentation_mask.astype(np.uint8))
        mask_images.append(masked_image)

    valid_images, valid_mask_indices = classify_images(
        model=model,
        label_mapping=label_mapping,
        images=mask_images,
        target_class=target_class,
        confidence_threshold=confidence_threshold
    )
    
    if len(valid_images) == 0:
        print(f"No valid masks for class '{target_class}' in image: {image_path}")
        return
    
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    valid_masks = [masks[i] for i in valid_mask_indices]
    show_anns(valid_masks, image)
    plt.axis('off')
    plt.show()

device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="F:\\Studium\\Master\\Thesis\\extRepos\\sam\\segment-anything\\chkpts\\sam_vit_h_4b8939.pth")
sam.to(device=device)

# Maskengenerator initialisieren
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=4,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

model = models.mobilenet_v2(pretrained=False)
num_classes = 15
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("./chkpt/mobilenetv2_finetuned_4.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

label_mapping = {
    0: "A-Building",
    1: "B-Building",
    2: "C-Building",
    3: "E-Building",
    4: "F-Building",
    5: "G-Building",
    6: "H-Building",
    7: "I-Building",
    8: "L-Building",
    9: "M-Building",
    10: "N-Building",
    11: "O-Building",
    12: "R-Building",
    13: "Z-Building",
    14: "other"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single image and filter masks by building class.")
    
    # Argument für den Gebäudenamen (erforderlich)
    parser.add_argument(
        "--building_class", 
        type=str, 
        required=True,
        help="The name of the building class to keep (e.g., 'A-Building')"
    )

    # Argument für den Bildpfad (erforderlich)
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="Path to the image to be processed"
    )

    args = parser.parse_args()

    target_class = args.building_class
    image_path = args.image_path

    # Überprüfen, ob der Gebäudename gültig ist
    if target_class not in label_mapping.values():
        print(f"Invalid building class: {target_class}")
        exit(1)

    # Überprüfen, ob das Bild existiert
    if not os.path.isfile(image_path):
        print(f"Invalid image path: {image_path}")
        exit(1)

    # Bild verarbeiten
    process_image(image_path, mask_generator, model, label_mapping, target_class)