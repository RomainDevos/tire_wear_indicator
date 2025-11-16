# ============================================================
# dataset.py
# Dataset VOC pour les témoins + remapping 95→90
# ============================================================

import os
import glob
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image


# ============================================================
# Classes finales (temoin:95 → temoin:90)
# ============================================================

CLASS_NAMES = [
    "temoin:0",
    "temoin:25",
    "temoin:50",
    "temoin:75",
    "temoin:80",
    "temoin:90",
    "temoin:100",
]

CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}
# => 1..7 (0 sera le background)


def normalize_label(raw_label: str) -> str:
    """Remappe temoin:95 → temoin:90."""
    if raw_label == "temoin:95":
        return "temoin:90"
    return raw_label


# ============================================================
# Dataset VOC
# ============================================================

class TemoinVOCDataset(Dataset):

    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms

        self.annotation_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
        if len(self.annotation_files) == 0:
            raise RuntimeError(f"Aucun fichier XML dans {annotations_dir}")

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        xml_path = self.annotation_files[idx]
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        img_path = os.path.join(self.images_dir, filename)

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for obj in root.findall("object"):
            raw = obj.find("name").text
            label = normalize_label(raw)

            if label not in CLASS_NAME_TO_ID:
                print(f"[WARN] label inconnu ignoré : {label}")
                continue

            label_id = CLASS_NAME_TO_ID[label]

            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)

        if len(boxes) == 0:
            raise RuntimeError(f"Pas de box dans {xml_path}")

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
