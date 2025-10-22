
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from pycocotools.coco import COCO

class COCODataset(Dataset):
    """COCO format dataset returning (image_tensor, target_dict)."""
    def __init__(self, root_dir, annotation_file, transforms=None, min_area=0.0):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = [img_id for img_id in self.coco.imgs.keys() if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_idx = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}  # 0 is background
        self.class_names = [c['name'] for c in self.coco.loadCats(self.cat_ids)]
        self.transforms = transforms
        self.min_area = min_area
        print(f"COCO dataset loaded: {len(self.image_ids)} images, {len(self.class_names)} classes")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            if ann['area'] < self.min_area:
                continue
            x, y, bw, bh = ann['bbox']
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            if (x2 - x1) * (y2 - y1) < self.min_area:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_idx[ann['category_id']])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        if self.transforms:
            transformed = self.transforms(image=np.array(image), bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.uint8)
        }
        return image, target
