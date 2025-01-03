import numpy as np
import torch
import clip
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def zeroshot_classifier(classnames):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = clip.tokenize(classname).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
    model.load_state_dict(torch.load("./chkpt/clip_b16_just_names.pt", map_location=device, weights_only=True))
    model.to(device)

    classes = ["A-Building", "B-Building", "C-Building", "E-Building", "F-Building", "G-Building", "H-Building", "I-Building", "J-Building", "L-Building", "M-Building", "N-Building", "O-Building", "R-Building", "Z-Building", "cube", "stuff"]

    custom_images_path = "./output_masks/"
    custom_dataset = CustomImageDataset(image_folder=custom_images_path, transform=preprocess)

    # Setze num_workers=0, um Multiprocessing zu vermeiden
    loader = DataLoader(custom_dataset, batch_size=32, shuffle=False, num_workers=2, timeout=10)

    zeroshot_weights = zeroshot_classifier(classes)

    confidence_threshold = 50  # Setze die Konfidenz-Schwelle

    with torch.no_grad():
        for images, img_paths in loader:
            images = images.cuda()

            # Predict image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute logits (image features similarity with class text embeddings)
            logits = 100. * image_features @ zeroshot_weights

            # Find the best-matching class for each image
            probs = logits.softmax(dim=-1)
            top_probs, top_labels = probs.topk(1, dim=-1)

            for img_path, top_label, top_prob in zip(img_paths, top_labels, top_probs):
                confidence = top_prob.item() * 100  # Umrechnung in Prozent
                if confidence < confidence_threshold:
                    # Lösche das Bild, da die Konfidenz unter der Schwelle liegt
                    print(f"Deleting {img_path} due to low confidence: {confidence:.2f}%")
                    os.remove(img_path)
                else:
                    predicted_class = classes[top_label.item()]
                    print(f"Image: {img_path} -> Predicted Class: {predicted_class} with confidence: {confidence:.2f}%")
