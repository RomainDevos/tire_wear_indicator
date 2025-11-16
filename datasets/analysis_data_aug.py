import os
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt

# === PARAMÈTRES ===
# (chemin de ton dossier augmenté)
dataset_dir = r"C:\Users\romai\Documents\Michelin\drive\data_tu\data_tu\image_balanced"

# === COMPTE DES CLASSES ===
class_counts = Counter()

# Parcours tous les fichiers XML du dossier
xml_files = [f for f in os.listdir(dataset_dir) if f.endswith(".xml")]

if not xml_files:
    print(f" Aucun fichier XML trouvé dans : {dataset_dir}")
    exit()

for xml_file in xml_files:
    xml_path = os.path.join(dataset_dir, xml_file)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f" Erreur de lecture : {xml_file}")
        continue

    # Extraction des classes (balises <name>)
    for obj in root.findall("object"):
        cname = obj.find("name").text.strip()
        class_counts[cname] += 1

# === AFFICHAGE TEXTE ===
print("\n===  RÉPARTITION DES CLASSES APRÈS AUGMENTATION ===")
total_objects = sum(class_counts.values())

for cls, count in sorted(class_counts.items(), key=lambda x: x[0]):
    print(f"{cls:<15}: {count:5d}")

print(f"\n Total d'objets annotés : {total_objects}")
print(f" Nombre de classes uniques : {len(class_counts)}")

# === VISUALISATION ===
plt.figure(figsize=(8, 4))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Distribution dess classes après augmentation")
plt.xlabel("Classes")
plt.ylabel("Nombre d'objets")
plt.tight_layout()
plt.show()
