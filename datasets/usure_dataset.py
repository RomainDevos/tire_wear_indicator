
import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class UsureDataset(Dataset):
    """VOC-style XML dataset used in original script."""
    def __init__(self, list_file, images_dir, transforms=None, class_names=None, min_area=0.0):
        with open(list_file, 'r', encoding='utf-8') as f:
            self.image_names = [l.strip() for l in f if l.strip()]
        self.images_dir = images_dir
        self.transforms = transforms
        self.min_area = min_area

        # Auto-découverte des classes
        if class_names is None:
            labels = set()
            for name in self.image_names:
                xml_path = os.path.join(images_dir, f"{name}.xml")
                if os.path.isfile(xml_path):
                    root = ET.parse(xml_path).getroot()
                    for obj in root.findall('object'):
                        n = obj.findtext('name')
                        if n:
                            labels.add(n.strip())
            self.class_names = sorted(labels)
        else:
            self.class_names = class_names

        self.cls_to_idx = {c: i + 1 for i, c in enumerate(self.class_names)}
        print(f"Usure dataset loaded: {len(self.image_names)} images, {len(self.class_names)} classes")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, f"{name}.jpg")
        xml_path = os.path.join(self.images_dir, f"{name}.xml")
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        boxes, labels, areas, iscrowd = [], [], [], []
        if os.path.isfile(xml_path):
            root = ET.parse(xml_path).getroot()
            for obj in root.findall('object'):
                cname = obj.findtext('name').strip()
                if cname not in self.cls_to_idx:
                    continue
                bnd = obj.find('bndbox')
                xmin, ymin, xmax, ymax = [float(bnd.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.cls_to_idx[cname])
                areas.append((xmax - xmin) * (ymax - ymin))
                iscrowd.append(0)

        if self.transforms:
            transformed = self.transforms(image=np.array(image), bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.uint8)
        }
        return image, target
