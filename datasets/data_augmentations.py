import os
import cv2
import random
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

import albumentations as A
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

# =====================================================================
# ARGUMENTS & CHEMINS PAR DÉFAUT
# =====================================================================

parser = argparse.ArgumentParser(
    description="Augmentation équilibrée avancée avec anti-biais + analyse TSNE"
)

parser.add_argument(
    "--input-dir",
    type=str,
    default=r"C:\Users\romai\Documents\Michelin\drive\data_tu\data_tu\image_base",
    help="Dossier contenant les images et XML"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default=r"C:\Users\romai\Documents\Michelin\drive\data_tu\data_tu\image_balanced",
    help="Dossier où stocker les images augmentées"
)

parser.add_argument(
    "--target",
    type=int,
    default=700,
    help="Nombre d'images souhaité par classe"
)

parser.add_argument(
    "--max-per-image",
    type=int,
    default=25,
    help="Limite d'augmentations par image pour éviter les biais"
)

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
TARGET = args.target
MAX_PER_IMAGE = args.max_per_image

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# PIPELINE D'AUGMENTATION
# =====================================================================

augment = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.4),
        A.Affine(
            scale=(0.8, 1.2),
            rotate=(-15, 15),
            shear=(-5, 5),
            translate_percent=(-0.05, 0.05),
            p=0.6,
        ),
        A.Perspective(scale=(0.03, 0.06), p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomShadow(p=0.3),
        A.RandomGamma(p=0.4),
        A.GaussNoise(p=0.3),   # Correction: var_limit retiré (plus compatible)
        A.HueSaturationValue(p=0.3),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"])
)

# =====================================================================
# ANALYSE DU DATASET ORIGINAL
# =====================================================================

print("\nAnalyse du dataset :", INPUT_DIR)

images = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

class_files = defaultdict(list)

for img_name in images:
    xml_path = os.path.join(INPUT_DIR, os.path.splitext(img_name)[0] + ".xml")
    if not os.path.exists(xml_path):
        continue

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print("XML invalide, ignoré :", xml_path)
        continue

    root = tree.getroot()

    first_obj = root.find("object")
    if first_obj is None:
        continue

    cls = first_obj.find("name").text.strip()
    class_files[cls].append((img_name, xml_path))

print("\n=== DISTRIBUTION ORIGINALE ===")
for cls, files in class_files.items():
    print(f"{cls:<12}: {len(files)} images")

# =====================================================================
# AUGMENTATION EQUILIBRÉE AVEC ANTI-BIAIS
# =====================================================================

print(
    f"\nGénération pour atteindre {TARGET} images par classe "
    f"(max {MAX_PER_IMAGE} augmentations par image)...\n"
)

for cls, file_list in class_files.items():
    current = len(file_list)
    missing = TARGET - current

    if missing <= 0:
        print(f"{cls} OK ({current}) — pas d'augmentation nécessaire.")
        continue

    print(f"Classe {cls} : {current} → {TARGET} (à générer = {missing})")

    # compteur par image
    usage_counter = {img: 0 for img, _ in file_list}

    for i in tqdm(range(missing), desc=f"Augmenting {cls}"):

        candidates = [
            (img, xml)
            for (img, xml) in file_list
            if usage_counter[img] < MAX_PER_IMAGE
        ]

        if not candidates:
            print(f"Plus aucune image utilisable pour {cls}.")
            break

        img_name, xml_path = random.choice(candidates)
        usage_counter[img_name] += 1

        img = cv2.imread(os.path.join(INPUT_DIR, img_name))
        if img is None:
            print("Impossible de lire l'image :", img_name)
            continue

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            print("XML invalide :", xml_path)
            continue

        root = tree.getroot()

        bboxes, labels = [], []
        for obj in root.findall("object"):
            label = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        if not bboxes:
            continue

        transformed = augment(image=img, bboxes=bboxes, labels=labels)
        new_img = transformed["image"]
        new_bboxes = transformed["bboxes"]

        # Nettoyage du nom : Windows interdit ':'
        safe_cls = cls.replace(":", "_")

        new_img_name = f"{safe_cls}_aug_{i}.jpg"
        new_xml_name = f"{safe_cls}_aug_{i}.xml"

        out_img_path = os.path.join(OUTPUT_DIR, new_img_name)
        out_xml_path = os.path.join(OUTPUT_DIR, new_xml_name)

        cv2.imwrite(out_img_path, new_img)

        # Mise à jour XML
        for obj, bb in zip(root.findall("object"), new_bboxes):
            bbox = obj.find("bndbox")
            bbox.find("xmin").text = str(int(bb[0]))
            bbox.find("ymin").text = str(int(bb[1]))
            bbox.find("xmax").text = str(int(bb[2]))
            bbox.find("ymax").text = str(int(bb[3]))

        filename_node = root.find("filename")
        if filename_node is not None:
            filename_node.text = new_img_name

        tree.write(out_xml_path)

print("\nDataset équilibré généré dans :", OUTPUT_DIR)

# =====================================================================
# ANALYSE TSNE
# =====================================================================

print("\nAnalyse TSNE du dataset généré...")

all_images = []
all_labels = []

jpg_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".jpg")]

MAX_TSNE = 2000
if len(jpg_files) > MAX_TSNE:
    jpg_files = random.sample(jpg_files, MAX_TSNE)

for f in jpg_files:
    try:
        img = Image.open(os.path.join(OUTPUT_DIR, f)).convert("L").resize((48, 48))
    except:
        continue

    all_images.append(np.array(img).flatten())
    all_labels.append(f.split("_")[0])

if len(all_images) > 10:
    all_images = np.array(all_images)

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            init="random",
            random_state=42
        )
        embedding = tsne.fit_transform(all_images)

        plt.figure(figsize=(10, 8))
        for cls in sorted(set(all_labels)):
            idx = [i for i, l in enumerate(all_labels) if l == cls]
            plt.scatter(embedding[idx, 0], embedding[idx, 1], s=10, label=cls)

        plt.legend()
        plt.title("TSNE du dataset équilibré")

        out_tsne = os.path.join(OUTPUT_DIR, "tsne_analysis.png")
        plt.savefig(out_tsne, dpi=150)
        plt.close()

        print("TSNE sauvegardé sous :", out_tsne)

    except Exception as e:
        print("Erreur TSNE :", e)
else:
    print("Pas assez d'images pour un TSNE.")

# =====================================================================
# PANEL VISUEL
# =====================================================================

print("\nGénération d'un panel visuel...")

jpg_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".jpg")]

if len(jpg_files) >= 16:
    samples = random.sample(jpg_files, 16)
else:
    samples = jpg_files

if samples:
    panel = Image.new("RGB", (800, 800), "white")

    for i, f in enumerate(samples[:16]):
        try:
            img = Image.open(os.path.join(OUTPUT_DIR, f)).resize((200, 200))
        except:
            continue

        x = (i % 4) * 200
        y = (i // 4) * 200
        panel.paste(img, (x, y))

    out_panel = os.path.join(OUTPUT_DIR, "panel_preview.jpg")
    panel.save(out_panel)

    print("Panel visuel sauvegardé sous :", out_panel)

print("\nTraitement terminé.")
