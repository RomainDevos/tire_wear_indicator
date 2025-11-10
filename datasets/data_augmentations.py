import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
import argparse

# === PARSEUR D'ARGUMENTS ===
parser = argparse.ArgumentParser(description="Augmentation d'images et annotations XML (Pascal VOC) avec Albumentations")

parser.add_argument(
    "--input-dir",
    type=str,
    default=r"C:\Users\romai\Documents\Michelin\drive\data_tu\data_tu\image_base",
    help="Chemin vers le dossier contenant les images et XML"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=r"C:\Users\romai\Documents\Michelin\drive\data_tu\data_tu\image_aug",
    help="Dossier de sortie pour enregistrer les résultats"
)
parser.add_argument("--n-aug", type=int, default=3, help="Nombre d'augmentations à générer par image")
parser.add_argument("--resize", type=int, default=None, help="Taille de redimensionnement (ex: 300 pour 300x300)")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print(f" Dossier d'entrée : {input_dir}")
print(f" Dossier de sortie : {output_dir}")

# === PIPELINE D’AUGMENTATION ===
transform_list = [
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.GaussNoise(p=0.3)
]
if args.resize:
    transform_list.append(A.Resize(args.resize, args.resize))

transform = A.Compose(transform_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# === BOUCLE SUR LES IMAGES ===
files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"🔍 {len(files)} fichiers trouvés dans le dossier d'entrée.")

if len(files) == 0:
    print(" Aucun fichier image trouvé dans le dossier spécifié. Vérifie le chemin et les extensions.")
    exit()

for file in files:
    img_path = os.path.join(input_dir, file)

    # Correspondance XML (remplace .jpg ou .jpeg par .xml aussi)
    xml_path = os.path.splitext(img_path)[0] + ".xml"

    if not os.path.exists(xml_path):
        print(f" Pas d’annotation pour {file}, ignoré.")
        continue

    # --- Lecture image ---
    image = cv2.imread(img_path)
    if image is None:
        print(f" Impossible de lire {file}. Fichier ignoré.")
        continue

    # --- Lecture XML et extraction des bounding boxes ---
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes, labels = [], []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    if not bboxes:
        print(f" Aucune bounding box dans {file}, ignoré.")
        continue

    # --- Générer plusieurs augmentations ---
    for i in range(args.n_aug):
        transformed = transform(image=image, bboxes=bboxes, labels=labels)
        aug_img = transformed["image"]
        aug_bboxes = transformed["bboxes"]

        # --- Sauvegarde image augmentée ---
        new_img_name = os.path.splitext(file)[0] + f"_aug{i}.png"
        new_img_path = os.path.join(output_dir, new_img_name)
        cv2.imwrite(new_img_path, aug_img)

        # --- Mise à jour XML ---
        for obj, new_bbox in zip(root.findall("object"), aug_bboxes):
            bbox = obj.find("bndbox")
            bbox.find("xmin").text = str(int(new_bbox[0]))
            bbox.find("ymin").text = str(int(new_bbox[1]))
            bbox.find("xmax").text = str(int(new_bbox[2]))
            bbox.find("ymax").text = str(int(new_bbox[3]))

        root.find("filename").text = new_img_name
        new_xml_name = os.path.splitext(file)[0] + f"_aug{i}.xml"
        tree.write(os.path.join(output_dir, new_xml_name))

        print(f"✅ {file} → {new_img_name}")

print(f"\n Augmentation terminée ! Les fichiers sont dans : {output_dir}")
