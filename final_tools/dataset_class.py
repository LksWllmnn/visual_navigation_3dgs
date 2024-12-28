import os
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import Compose, Resize, ToTensor
import clip
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# Farbwerte der Gebäude als Hex zu RGB
BUILDING_COLORS = {
    "A-Building": (0, 0, 255, 255),
    "B-Building": (0, 255, 0, 255),
    "C-Building": (255, 0, 0, 255),
    "E-Building": (255, 255, 255, 255),
    "F-Building": (255, 235, 4, 255),
    "G-Building": (128, 128, 128, 255),
    "H-Building": (255, 32, 98, 255),
    "I-Building": (255, 25, 171, 255),
    "M-Building": (255, 73, 101, 255),
    "N-Building": (145, 255, 114, 255),
    "L-Building": (93, 71, 255, 255),
    "O-Building": (153, 168, 255, 255),
    "R-Building": (64, 0, 75, 255),
    "Z-Building": (18, 178, 0, 255),
}

BACKGROUND_COLOR = (0, 0, 0, 255)

# Preprocessing for images
preprocess = Compose([
    Resize((224, 224)),  # Resize to CLIP model input size
    ToTensor()           # Convert to tensor
])

class ImageTitleDataset:
    # def __init__(self, root_dir, transform=None, device='cuda', filter_images = False, type="clip"):
    #     self.image_paths = []
    #     self.labels = []
    #     self.original_labels = []  # Ursprüngliche String-Labels speichern
    #     self.valid_images = []
    #     self.invalid_images = []
    #     self.transform = transform
    #     self.type = type
    #     self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    #     # Store count of valid images per building
    #     self.building_counts = Counter()

    #     # Iterate over building folders
    #     building_folders = [b for b in BUILDING_COLORS.keys() if os.path.isdir(os.path.join(root_dir, b))]
    #     print(building_folders)

    #     for building in tqdm(building_folders, desc="Processing buildings"):
    #         color = torch.tensor(BUILDING_COLORS[building], device=self.device, dtype=torch.uint8)
    #         building_dir = os.path.join(root_dir, building)
    #         print(building_dir)
    #         building_valid_images = []

    #         for filename in tqdm(os.listdir(building_dir), desc=f"Filtering images in {building}", leave=False):
    #             if filename.endswith("camera.png"):
    #                 base_name = filename.replace(".camera.png", "")
    #                 seg_path = os.path.join(building_dir, f"{base_name}.camera.semantic segmentation.png")

    #                 if not os.path.exists(seg_path):
    #                     continue

    #                 # Check if the semantic segmentation image matches the conditions
    #                 full_image_path = os.path.join(building_dir, filename)
    #                 if filter_images:
    #                     if self.is_valid_image(seg_path, color):
    #                         building_valid_images.append(full_image_path)
    #                         self.building_counts[building] += 1
    #                     else:
    #                         self.invalid_images.append(full_image_path)
    #                 else:
    #                     building_valid_images.append(full_image_path)
    #                     self.building_counts[building] += 1

    #         # Limit to 500 valid images per building
    #         selected_images = building_valid_images[:500]
    #         self.image_paths.extend(selected_images)
    #         #self.labels.extend([building] * len(selected_images))
    #         self.valid_images.extend(selected_images)
    #         self.original_labels.extend([building] * len(selected_images))  # Originale Labels speichern
    #     self.label_encoder = LabelEncoder()
    #     #self.labels = self.label_encoder.fit_transform(self.labels)
    #     self.original_labels.extend([building] * len(selected_images))  # Originale Labels speichern

    def __init__(self, root_dir, device='cuda', model_type = "clip", transform=None, filter=False):
        self.image_paths = []
        self.labels = []
        self.valid_images = []
        self.invalid_images = []
        self.filter = filter
        self.transform = transform
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Store count of valid images per building
        self.building_counts = Counter()

        # Iterate over building folders
        building_folders = [b for b in BUILDING_COLORS.keys() if os.path.isdir(os.path.join(root_dir, b))]

        for building in tqdm(building_folders, desc="Processing buildings"):
            color = torch.tensor(BUILDING_COLORS[building], device=self.device, dtype=torch.uint8)
            building_dir = os.path.join(root_dir, building)

            building_valid_images = []

            for filename in tqdm(os.listdir(building_dir), desc=f"Filtering images in {building}", leave=False):
                if filename.endswith("camera.png"):
                    base_name = filename.replace(".camera.png", "")
                    seg_path = os.path.join(building_dir, f"{base_name}.camera.semantic segmentation.png")

                    if not os.path.exists(seg_path):
                        continue

                    # Check if the semantic segmentation image matches the conditions
                    full_image_path = os.path.join(building_dir, filename)
                    if self.filter:
                        if self.is_valid_image(seg_path, color):
                            building_valid_images.append(full_image_path)
                            self.building_counts[building] += 1
                        else:
                            self.invalid_images.append(full_image_path)
                    else:
                        building_valid_images.append(full_image_path)
                        self.building_counts[building] += 1


            # Limit to 500 valid images per building
            selected_images = building_valid_images[:500]
            self.image_paths.extend(selected_images)
            self.labels.extend([building] * len(selected_images))
            self.valid_images.extend(selected_images)

    def is_valid_image(self, seg_path, target_color):
        """Check if the segmentation image is valid for the target building."""
        seg_image = Image.open(seg_path).convert("RGBA")
        seg_array = torch.tensor(np.array(seg_image), device=self.device, dtype=torch.uint8)

        # Count pixel occurrences, excluding the background color
        unique, counts = torch.unique(seg_array.view(-1, 4), dim=0, return_counts=True)
        color_counts = {tuple(color.tolist()): count.item() for color, count in zip(unique, counts) if tuple(color.tolist()) != BACKGROUND_COLOR}

        # Exclude if target color is not present
        if tuple(target_color.tolist()) not in color_counts:
            return False

        return True

    def __len__(self):
        # if(self.model_type == "clip"):
        #     return len(self.original_labels)
        # else:
        #     return len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if self.model_type == "clip":
            image = preprocess(Image.open(self.image_paths[idx]).convert("RGB"))
            label = clip.tokenize(self.labels[idx])[0]  # Tokenize the label
            #label = clip.tokenize(self.original_labels[idx])[0]  # Originale Labels tokenisieren
        elif self.model_type == "resnet":
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)
        return image, label

if __name__ == "__main__":
    root_directory = r"F:\Studium\Master\Thesis\data\perception\usefull_data\finetune_data\building_big_surround_pictures"  # Replace with your root directory
    dataset = ImageTitleDataset(root_dir=root_directory, filter_images=True)

    # Print dataset statistics
    print("Dataset statistics:")
    for building, count in dataset.building_counts.items():
        print(f"{building}: {count} images")

    print(f"Total images: {len(dataset)}")

    # Plot 20 valid and 20 invalid images
    def plot_images(image_paths, title, num_images=20):
        plt.figure(figsize=(15, 10))
        for i, img_path in enumerate(image_paths[:num_images]):
            img = Image.open(img_path)
            plt.subplot(4, 5, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path))
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    print("\nDisplaying valid images:")
    plot_images(dataset.valid_images, "Valid Images")

    print("\nDisplaying invalid images:")
    plot_images(dataset.invalid_images, "Invalid Images")
