import os
import shutil
from PIL import Image

def copy_rename_and_resize_images(source_folder, destination_folder):
    # Sicherstellen, dass der Zielordner existiert
    os.makedirs(destination_folder, exist_ok=True)
    
    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            # Neuen Namen erstellen, indem "semantic segmentation" entfernt wird
            new_name = filename.replace(".semantic segmentation", "").strip()
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, new_name)
            
            # Bild öffnen und Größe anpassen
            with Image.open(source_path) as img:
                resized_img = img.resize((799, 599))  # Größe ändern
                resized_img.save(destination_path)
            
            print(f"Verarbeitet: {source_path} → {destination_path} (Größe angepasst)")

# Beispielaufruf
source_folder = r"F:\Studium\Master\Thesis\data\perception\usefull_data\train_data\labels"  # Pfad zum Quellordner
destination_folder = r"F:\Studium\Master\Thesis\data\perception\usefull_data\lable-gs\images"  # Pfad zum Zielordner

copy_rename_and_resize_images(source_folder, destination_folder)