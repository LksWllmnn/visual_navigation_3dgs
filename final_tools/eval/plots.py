import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Liste mit Dateinamen resnet
# file_paths = [
#     r"F:\Studium\Master\Thesis\data\final_final_results\resnet_big-surround\just-mask\output_log.csv",      #big-surround
#     r"F:\Studium\Master\Thesis\data\final_final_results\resnet_scene\just-mask\output_log.csv",             #scene 
#     r"F:\Studium\Master\Thesis\data\final_final_results\resnet_surround\just-mask\output_log.csv"]          #surround

# # lerf lite
# file_paths = [
#     r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\no-finetuning\just-mask\output_log.csv",     #no-finetuning
#     r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\big-surround\just-mask\output_log.csv",      #big-surround
#     r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\scene\just-mask\output_log.csv",             #scene 
#     r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\surround\just-mask\output_log.csv"]          #surround

# Mask R-CNN
file_paths = [
    r"F:\Studium\Master\Thesis\data\final_final_results\mrcnn_big-surround\just-mask\output_log.csv",      #big-surround
    r"F:\Studium\Master\Thesis\data\final_final_results\mrcnn_scene\just-mask\output_log.csv",             #scene 
    r"F:\Studium\Master\Thesis\data\final_final_results\mrcnn_surround\just-mask\output_log.csv"]          #surround

dataframes = [pd.read_csv(file, sep=",") for file in file_paths]
fig, ax = plt.subplots(1, len(dataframes), figsize=(20, 6), sharey=True)

for i, df in enumerate(dataframes):
    print(f"Spaltennamen der Datei {i+1}: {df.columns}")
    df[" IoU"] = pd.to_numeric(df[" IoU"], errors='coerce')
    
    # Filtern der DataFrames, bei denen die Kategorie "Richtig" vorhanden ist oder "Falsch" nicht vorhanden
    if " Category" in df.columns:
        df_filtered = df[df[" Category"].isin([" Richtig vorhanden", " Falsch nicht vorhanden"])]
        print("hi")
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

        # if i == 0:
        #     ax[i].set_title(f"No Finetuning (Mean: {mean: .5f}, Count: {count}, Median: {median: .5f})")
        # if i == 1:
        #     ax[i].set_title(f"Big-Surround (Mean: {mean: .5f}, Count: {count}, Median: {median: .5f})")
        # if i == 2:
        #     ax[i].set_title(f"Scene (Mean: {mean: .5f}, Count: {count}, Median: {median: .5f})")
        # if i == 3:
        #     ax[i].set_title(f"Surround (Mean: {mean: .5f}, Count: {count}, Median: {median: .5f})")
        
        if i == 0:
            ax[i].set_title(f"Big-Surround (Mean: {mean: .5f}, Count: {count}, Median: {median: .5f})")
        if i == 1:
            ax[i].set_title(f"Scene (Mean: {mean: .5f}, Count: {count}, Median: {median: .5f})")
        if i == 2:
            ax[i].set_title(f"Surround (Mean: {mean: .5f}, Count: {count}, Median: {median: .5f})")
        
        ax[i].set_ylabel("IoU" if i == 0 else "")
        ax[i].set_xlabel("Boxplot")

plt.tight_layout()
plt.savefig("boxplots_iou.png")
plt.close()

# Kategorien definieren (damit wir immer in dieser Reihenfolge zählen)
categories = [" Falsch vorhanden", " Richtig vorhanden", " Richtig nicht vorhanden", " Falsch nicht vorhanden"]

# Daten einlesen und die Häufigkeit der Kategorien zählen
category_counts = []
for i, df in enumerate(dataframes):
    # Zähle, wie oft jede Kategorie vorkommt
    counts = df[" Category"].value_counts()

    # Stelle sicher, dass alle Kategorien vorkommen, auch wenn eine fehlt (mit 0 füllen)
    counts = counts.reindex(categories, fill_value=0)

    # Benennen der Serie nach der Datei
    # if i == 0:
    #     counts.name = "No-Finetuning"
    # elif i == 1:
    #     counts.name = "Big-Surround"
    # elif i == 2:
    #     counts.name = "Scene"
    # elif i == 3:
    #     counts.name = "Surround"
    if i == 0:
        counts.name = "Big-Surround"
    elif i == 1:
        counts.name = "Scene"
    elif i == 2:
        counts.name = "Surround"

    # Füge die gezählten Werte zur Liste hinzu
    category_counts.append(counts)

# Erstelle das DataFrame aus den gezählten Werten
category_df = pd.DataFrame(category_counts).T  # .T für Transponieren (Dateien als Spalten)

# Zeige das DataFrame an, um sicherzustellen, dass es korrekt erstellt wurde
print(category_df)

# Erstelle das Balkendiagramm
category_df.plot(kind="barh", figsize=(10, 6), subplots=True)

# Titel und Achsenbeschriftungen hinzufügen
plt.title("Kategorie-Auswertung pro Datei")
plt.ylabel("Anzahl")
plt.xlabel("Dateien")
plt.legend(title="Kategorien")

# Layout optimieren und Bild speichern
plt.tight_layout()
plt.savefig("categories_count_per_file.png")
plt.close()

# Ausgabe für das Debugging
print("Plots wurden erstellt und gespeichert: categories_count_per_file.png")