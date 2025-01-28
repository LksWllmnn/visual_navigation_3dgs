import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import precision_score, recall_score, f1_score

output_folder = r"F:\Studium\Master\Thesis\data\final_final_results\0_plots"
# Erstelle den Ordner, falls er nicht existiert
os.makedirs(output_folder, exist_ok=True)

def map_filename(number): 
    if number == 0:
            return "No Finetuning"
    if number == 1:
        return "Big-Surround"
    if number == 2:
        return "Scene"
    if number == 3:
        return "Surround"


# Liste mit Dateinamen resnet
# file_paths = [
#     r"F:\Studium\Master\Thesis\data\final_final_results\resnet_big-surround\just-mask\output_log.csv",
#     r"F:\Studium\Master\Thesis\data\final_final_results\resnet_big-surround\just-mask\output_log.csv",      #big-surround
#     r"F:\Studium\Master\Thesis\data\final_final_results\resnet_scene\just-mask\output_log.csv",             #scene 
#     r"F:\Studium\Master\Thesis\data\final_final_results\resnet_surround\just-mask\output_log.csv"]          #surround
# tech="resnet"

# # lerf lite
# file_paths = [
#     r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\no-finetuning\just-mask\output_log.csv",     #no-finetuning
#     r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\big-surround\just-mask\output_log.csv",      #big-surround
#     r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\scene\just-mask\output_log.csv",             #scene 
#     r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\surround\just-mask\output_log.csv"]          #surround
# tech="lerf-lite"

# Mask R-CNN
# file_paths = [
#     r"F:\Studium\Master\Thesis\data\final_final_results\mrcnn_no-finetuning\just-mask\output_log.csv",
#     r"F:\Studium\Master\Thesis\data\final_final_results\mrcnn_big-surround\just-mask\output_log.csv",      #big-surround
#     r"F:\Studium\Master\Thesis\data\final_final_results\mrcnn_scene\just-mask\output_log.csv",             #scene 
#     r"F:\Studium\Master\Thesis\data\final_final_results\mrcnn_surround\just-mask\output_log.csv"]          #surround
# tech="mask-r-cnn"

# Feature Splatting
file_paths = [
    r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\feature-splatting\no-finetuning\just_mask\output_log.csv",     #no-finetuning
    r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\feature-splatting\big-surround\just_mask\output_log.csv",      #big-surround
    r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\feature-splatting\scene\just_mask\output_log.csv",             #scene 
    r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\feature-splatting\surround\just_mask\output_log.csv"]
tech="feature-splatting"

dataframes = [pd.read_csv(file, sep=",") for file in file_paths]
fig, ax = plt.subplots(1, len(dataframes), figsize=(20, 6), sharey=True)


for i, df in enumerate(dataframes):
    print(f"Spaltennamen der Datei {i+1}: {df.columns}")
    df[" IoU"] = pd.to_numeric(df[" IoU"], errors='coerce')
    
    # Filtern der DataFrames, bei denen die Kategorie "Richtig" vorhanden ist oder "Falsch" nicht vorhanden
    if " Category" in df.columns:
        df_filtered = df[df[" Category"].isin([" Richtig vorhanden", " Falsch nicht vorhanden"])]
    else:
        df_filtered = df

    # Berechnung und Boxplot nur, wenn es gefilterte Daten gibt
    if not df_filtered.empty:
        print(len(df_filtered))
        median = df_filtered[" IoU"].median(numeric_only=True)
        mean = df_filtered[" IoU"].mean()
        count = df_filtered[" IoU"].count()
        
        # Boxplot erstellen
        sns.boxplot(y=df_filtered[" IoU"], ax=ax[i])
        ax[i].set_title(f"{map_filename(i)} (Mean: {mean: .3f}, Count: {count}, Median: {median: .3f})")
        
        ax[i].set_ylabel("IoU" if i == 0 else "")
        ax[i].set_xlabel("Boxplot")

plt.tight_layout()
boxplot_path = os.path.join(output_folder, f"{tech}_boxplots_iou.png")
plt.savefig(boxplot_path)
plt.close()


## Confusion matrix
dataframes = [pd.read_csv(file, sep=",") for file in file_paths]
# Funktion zum Mapping der Kategorien auf vorhanden/nicht vorhanden
def map_to_labels(category, is_true_label=True):
    """
    Map categories to 'vorhanden' or 'nicht vorhanden'.
    - `is_true_label`: Determines if the mapping is for true labels or predicted labels.
    """
    if is_true_label:
        if category in [" Richtig vorhanden", " Falsch nicht vorhanden"]:
            return "vorhanden"
        elif category in [" Richtig nicht vorhanden", " Falsch vorhanden"]:
            return "nicht vorhanden"
    else:
        if category in [" Richtig vorhanden", " Falsch vorhanden"]:
            return "vorhanden"
        elif category in [" Richtig nicht vorhanden", " Falsch nicht vorhanden"]:
            return "nicht vorhanden"
    return None  # Für den Fall, dass die Kategorie fehlt oder NaN ist

# Initialisiere leere Listen für die Labels
all_true_labels = []
all_predicted_labels = []

for i, file_path in enumerate(file_paths):
    # DataFrame laden
    df = pd.read_csv(file_path, sep=",")
    print(f"Verarbeite Datei {i+1}: {file_path}")

    # Überprüfen, ob die Kategorie-Spalte existiert
    if " Category" in df.columns:
        # Mapping der True Labels
        true_labels = df[" Category"].map(lambda x: map_to_labels(x, is_true_label=True))

        # Für Predicted Labels:
        # Hier nehmen wir dieselbe Spalte und wenden dieselbe Logik an, könnten aber andere Kriterien verwenden.
        predicted_labels = df[" Category"].map(lambda x: map_to_labels(x, is_true_label=False))

        # Fehlende Werte entfernen
        valid_indices = true_labels.notna() & predicted_labels.notna()
        true_labels = true_labels[valid_indices]
        predicted_labels = predicted_labels[valid_indices]

        # Konfusionsmatrix berechnen
        categories_mapped = ["vorhanden", "nicht vorhanden"]
        cm = confusion_matrix(true_labels, predicted_labels, labels=categories_mapped)

        # Konfusionsmatrix in DataFrame umwandeln
        cm_df = pd.DataFrame(cm, index=categories_mapped, columns=categories_mapped)

        precision = precision_score(true_labels, predicted_labels, labels=categories_mapped, average='binary', pos_label="vorhanden")
        recall = recall_score(true_labels, predicted_labels, labels=categories_mapped, average='binary', pos_label="vorhanden")
        f1 = f1_score(true_labels, predicted_labels, labels=categories_mapped, average='binary', pos_label="vorhanden")

        # Heatmap zeichnen
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.title(f'{map_filename(i)} - p:{precision: .3}, r:{recall: .3}, f1:{f1: .3}')

        # Speichere die Konfusionsmatrix als Bild
        file_name = f"{tech}_{map_filename(i)}_confusion_matrix.png"
        conf_matrix_path = os.path.join(output_folder, file_name)
        plt.savefig(conf_matrix_path)
        plt.close()

        # Ausgabe zur Bestätigung
        print(f"Konfusionsmatrix für Datei {map_filename(i)} wurde erstellt und gespeichert in: {conf_matrix_path}")