# import os
# import json
# from collections import defaultdict
# from tqdm import tqdm

# # Funktion, um die Instanzen aus einer JSON-Datei zu zählen
# def count_instances_in_json(json_file):
#     instance_counts = defaultdict(int)  # Zählt Instanzen pro Gebäude

#     try:
#         # JSON-Datei öffnen und laden
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         # Überprüfen, ob Instanzen vorhanden sind
#         if "captures" in data:
#             for capture in data["captures"]:
#                 if "annotations" in capture:
#                     for annotation in capture["annotations"]:
#                         if annotation.get("@type") == "type.unity.com/unity.solo.SemanticSegmentationAnnotation":
#                             instances = annotation.get("instances", [])
#                             if instances:
#                                 # Durch alle Instanzen iterieren und den labelName zählen
#                                 for instance in instances:
#                                     label_name = instance.get("labelName")
#                                     if label_name:
#                                         instance_counts[label_name] += 1

#     except (json.JSONDecodeError, UnicodeDecodeError) as e:
#         print(f"Fehler beim Laden der Datei {json_file}: {e}")

#     return instance_counts

# # Hauptprogramm
# if __name__ == "__main__":
#     root_dir = "C:\\Users\\Lukas\\AppData\\LocalLow\\DefaultCompany\\Fuwa_HDRP\\usefull_data\\solo_2"  # Ersetze mit deinem Pfad
#     building_counts = defaultdict(int)
#     json_count = 0  # Zähler für die gefundenen JSON-Dateien

#     # Gehe durch alle Gebäude im Hauptordner
#     buildings = os.listdir(root_dir)
    
#     with tqdm(total=len(buildings), desc="Processing buildings", unit="building") as building_bar:
#         for building in buildings:
#             building_path = os.path.join(root_dir, building)
#             if os.path.isdir(building_path):
#                 # Gehe durch alle Unterordner im Gebäudeordner (Sequenzen)
#                 sequence_folders = os.listdir(building_path)
#                 for sequence_folder in sequence_folders:
#                     # Path zur JSON-Datei für das aktuelle Bild
#                     #json_path = os.path.join(building_path, sequence_folder, "step0.frame_data.json")
#                     json_path = os.path.join(building_path, sequence_folder)
#                     # Überprüfen, ob es sich wirklich um eine JSON-Datei handelt
#                     if os.path.exists(json_path) and json_path.endswith(".json"):
#                         # Zähler für die gefundenen JSON-Dateien erhöhen
#                         json_count += 1

#                         # Instanzen aus der JSON-Datei zählen
#                         instance_counts = count_instances_in_json(json_path)
                        
#                         # Die gezählten Instanzen addieren
#                         for label, count in instance_counts.items():
#                             building_counts[label] += count

#                 building_bar.update(1)

#     # Die Anzahl der gefundenen JSON-Dateien ausgeben
#     print(f"Anzahl der gefundenen JSON-Dateien: {json_count}")

#     # Die Ergebnisse ausdrucken
#     if building_counts:
#         for building, count in building_counts.items():
#             print(f"Building {building} appeared {count} times in the dataset.")
#     else:
#         print("Keine Instanzen in den JSON-Dateien gefunden.")


import os
import json
from collections import defaultdict
from tqdm import tqdm

# Funktion, um die Instanzen aus einer JSON-Datei zu zählen
def count_instances_in_json(json_file):
    instance_counts = defaultdict(int)  # Zählt Instanzen pro Gebäude

    try:
        # JSON-Datei öffnen und laden
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Überprüfen, ob Instanzen vorhanden sind
        if "captures" in data:
            for capture in data["captures"]:
                if "annotations" in capture:
                    for annotation in capture["annotations"]:
                        if annotation.get("@type") == "type.unity.com/unity.solo.SemanticSegmentationAnnotation":
                            instances = annotation.get("instances", [])
                            if instances:
                                # Durch alle Instanzen iterieren und den labelName zählen
                                for instance in instances:
                                    label_name = instance.get("labelName")
                                    if label_name:
                                        instance_counts[label_name] += 1

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Fehler beim Laden der Datei {json_file}: {e}")

    return instance_counts

# Hauptprogramm
if __name__ == "__main__":
    root_dir = "C:\\Users\\Lukas\\AppData\\LocalLow\\DefaultCompany\\Fuwa_HDRP\\usefull_data\\solo_2"  # Ersetze mit deinem Pfad
    building_counts = defaultdict(int)
    json_count = 0  # Zähler für die gefundenen JSON-Dateien
    no_instances_count = 0  # Zähler für Bilder ohne Instanzen

    # Gehe durch alle Gebäude im Hauptordner
    buildings = os.listdir(root_dir)
    
    with tqdm(total=len(buildings), desc="Processing buildings", unit="building") as building_bar:
        for building in buildings:
            building_path = os.path.join(root_dir, building)
            if os.path.isdir(building_path):
                # Gehe durch alle Unterordner im Gebäudeordner (Sequenzen)
                sequence_folders = os.listdir(building_path)
                for sequence_folder in sequence_folders:
                    # Path zur JSON-Datei für das aktuelle Bild
                    json_path = os.path.join(building_path, sequence_folder)
                    # Überprüfen, ob es sich wirklich um eine JSON-Datei handelt
                    if os.path.exists(json_path) and json_path.endswith(".json"):
                        # Zähler für die gefundenen JSON-Dateien erhöhen
                        json_count += 1

                        # Instanzen aus der JSON-Datei zählen
                        instance_counts = count_instances_in_json(json_path)
                        
                        # Wenn keine Instanzen gefunden wurden, erhöhen wir den Zähler
                        if not instance_counts:
                            no_instances_count += 1
                        
                        # Die gezählten Instanzen addieren
                        for label, count in instance_counts.items():
                            building_counts[label] += count

                building_bar.update(1)

    # Die Anzahl der gefundenen JSON-Dateien ausgeben
    print(f"Anzahl der gefundenen JSON-Dateien: {json_count}")
    print(f"Anzahl der Bilder ohne Instanzen: {no_instances_count}")

    # Die Ergebnisse ausdrucken
    if building_counts:
        for building, count in building_counts.items():
            print(f"Building {building} appeared {count} times in the dataset.")
    else:
        print("Keine Instanzen in den JSON-Dateien gefunden.")
