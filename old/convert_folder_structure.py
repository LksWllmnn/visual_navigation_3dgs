import os
import shutil

# Quell- und Zielverzeichnisse anpassen
source_dir = r"C:\Users\Lukas\AppData\LocalLow\DefaultCompany\Fuwa_HDRP\singleBuildings_ResnetSam_2"
target_dir = r"C:\Users\Lukas\AppData\LocalLow\DefaultCompany\Fuwa_HDRP\Recordings_Single"

# Zielverzeichnis erstellen, falls nicht vorhanden
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Liste der Gebäude
buildings = [
    "A-Building", "B-Building", "C-Building", "E-Building", "F-Building", 
    "G-Building", "H-Building", "I-Building", "M-Building", "N-Building", 
    "L-Building", "O-Building", "R-Building", "Z-Building"
]

# Iteration über alle Gebäude
for building in buildings:
    print(f"Bearbeite Gebäude: {building}")

    # Pfad zum aktuellen Gebäude
    building_path = os.path.join(source_dir, building)
    
    # Überprüfen, ob das Gebäude-Verzeichnis existiert
    if os.path.exists(building_path):
        # Iteration über die "sequence.*"-Ordner
        for seq_folder in os.listdir(building_path):
            # Überprüfen, ob es sich um einen Sequenzordner handelt
            if seq_folder.startswith("sequence."):
                seq_path = os.path.join(building_path, seq_folder)
                # Sequenznummer extrahieren
                sequence_number = seq_folder.split('.')[-1]
                
                # Quelle für die Datei
                source_file = os.path.join(seq_path, "step0.camera.png")
                
                # Zieldatei definieren
                target_file = os.path.join(target_dir, f"Fuwa_{building}_{sequence_number}.png")
                
                # Überprüfen, ob die Quelldatei existiert
                if os.path.exists(source_file):
                    try:
                        # Datei kopieren
                        shutil.copy2(source_file, target_file)
                        print(f"Kopiert: {source_file} -> {target_file}")
                    except Exception as e:
                        print(f"Fehler beim Kopieren der Datei {source_file}: {e}")
                else:
                    print(f"Warnung: Datei fehlt: {source_file}")
    else:
        print(f"Warnung: Gebäudeordner fehlt: {building_path}")

print("Konvertierung abgeschlossen!")
