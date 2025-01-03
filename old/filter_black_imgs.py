import os
from PIL import Image
from tqdm import tqdm

def is_black_image(img_path, threshold=10):
    """
    Prüft, ob ein Bild fast vollständig schwarz ist.
    :param img_path: Pfad zum Bild
    :param threshold: Schwellenwert für Helligkeit
    :return: True, wenn das Bild schwarz ist, sonst False
    """
    with Image.open(img_path) as img:
        img = img.convert("L")  # In Graustufen konvertieren
        pixel_values = list(img.getdata())
        avg_pixel_value = sum(pixel_values) / len(pixel_values)
        return avg_pixel_value < threshold

def filter_and_save_images(input_dir, output_dir, threshold=10):
    """
    Filtert schwarze Bilder aus einem Eingabeverzeichnis und speichert die restlichen in einem Ausgabeverzeichnis.
    :param input_dir: Wurzelverzeichnis der Eingabedaten
    :param output_dir: Zielverzeichnis für die gefilterten Bilder
    :param threshold: Schwellenwert für Helligkeit
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    buildings = os.listdir(input_dir)
    print(f"Processing {len(buildings)} buildings...")

    with tqdm(total=len(buildings), desc="Filtering buildings", unit="building") as building_bar:
        for building in buildings:
            building_path = os.path.join(input_dir, building)
            if not os.path.isdir(building_path):
                continue
            
            # Zielverzeichnis für das aktuelle Gebäude
            target_building_path = os.path.join(output_dir, building)
            os.makedirs(target_building_path, exist_ok=True)

            for sequence_folder in os.listdir(building_path):
                image_path = os.path.join(building_path, sequence_folder, "step0.camera.png")
                if os.path.exists(image_path):
                    if not is_black_image(image_path, threshold):
                        # Zielpfad für das gefilterte Bild
                        target_sequence_path = os.path.join(target_building_path, sequence_folder)
                        os.makedirs(target_sequence_path, exist_ok=True)
                        target_image_path = os.path.join(target_sequence_path, "step0.camera.png")
                        
                        # Bild kopieren
                        img = Image.open(image_path)
                        img.save(target_image_path)

            building_bar.update(1)

if __name__ == "__main__":
    # Eingabe- und Ausgabe-Verzeichnisse
    input_dir = "C:/Users/Lukas/AppData/LocalLow/DefaultCompany/Fuwa_HDRP/singleBuildings_ResnetSam"
    output_dir = "C:/Users/Lukas/AppData/LocalLow/DefaultCompany/Fuwa_HDRP/filteredBuildings"

    # Schwellenwert für das Filtern schwarzer Bilder
    black_threshold = 10

    # Starte die Filterung und Speicherung
    filter_and_save_images(input_dir, output_dir, black_threshold)


