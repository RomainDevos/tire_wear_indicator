import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import json
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import _utils as model_utils
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights, SSDClassificationHead
from torchvision.models.vgg import VGG16_Weights
from tqdm import tqdm


# -----------------------------
# Datasets
# -----------------------------

class COCODataset(Dataset):
    """COCO format dataset returning (image_tensor, target_dict)."""
    def __init__(self, root_dir: str, annotation_file: str, transforms=None, min_area: float = 0.0):
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

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            if ann['area'] < self.min_area:
                continue
            x, y, bw, bh = ann['bbox']
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            # clamp
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            bw2 = x2 - x1
            bh2 = y2 - y1
            area = bw2 * bh2
            if area < self.min_area or bw2 < 2 or bh2 < 2:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_idx[ann['category_id']])
            areas.append(area)
            iscrowd.append(ann.get('iscrowd', 0))

        if len(boxes) == 0:
            # Skip images without usable boxes by picking another index (rare in filtered list)
            return self.__getitem__((idx + 1) % len(self))

        # Albumentations
        if self.transforms:
            transformed = self.transforms(image=np.array(image), bboxes=boxes, labels=labels)
            img_tensor = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            # Basic conversion
            from torchvision import transforms as T
            img_tensor = T.ToTensor()(image)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        areas_tensor = torch.tensor(areas, dtype=torch.float32)
        iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.uint8)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([image_id]),
            'area': areas_tensor,
            'iscrowd': iscrowd_tensor
        }
        return img_tensor, target


class UsureDataset(Dataset):
    """VOC-style XML dataset used in original script."""
    def __init__(self, list_file: str, images_dir: str, transforms=None, class_names: List[str] | None = None, min_area: float = 0.0):
        if not os.path.isfile(list_file):
            raise FileNotFoundError(list_file)
        if not os.path.isdir(images_dir):
            raise NotADirectoryError(images_dir)

        self.images_dir = images_dir
        with open(list_file, 'r', encoding='utf-8') as f:
            self.image_names = [l.strip() for l in f if l.strip()]
        self.transforms = transforms
        self.min_area = min_area

        if class_names is None:
            labels = set()
            for name in self.image_names:
                xml_path = os.path.join(images_dir, f"{name}.xml")
                if not os.path.isfile(xml_path):
                    continue
                try:
                    root = ET.parse(xml_path).getroot()
                    for obj in root.findall('object'):
                        n = obj.findtext('name')
                        if n:
                            labels.add(n.strip())
                except Exception:
                    continue
            self.class_names = sorted(labels)
        else:
            self.class_names = class_names
        self.cls_to_idx = {c: i + 1 for i, c in enumerate(self.class_names)}
        print(f"Usure dataset loaded: {len(self.image_names)} images, {len(self.class_names)} classes")

    def __len__(self):
        return len(self.image_names)

    def _parse_xml(self, xml_path: str, w: int, h: int):
        boxes, labels, areas, iscrowd = [], [], [], []

        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return boxes, labels, areas, iscrowd

        for obj in root.findall('object'):
            name = obj.findtext('name')
            if not name or name.strip() not in self.cls_to_idx:
                continue
            bnd = obj.find('bndbox')
            if bnd is None:
                continue

            try:
                xmin = float(bnd.findtext('xmin', '0'))
                ymin = float(bnd.findtext('ymin', '0'))
                xmax = float(bnd.findtext('xmax', '0'))
                ymax = float(bnd.findtext('ymax', '0'))
            except Exception:
                continue
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(xmin + 1, min(xmax, w))
            ymax = max(ymin + 1, min(ymax, h))
            bw = xmax - xmin
            bh = ymax - ymin
            area = bw * bh

            if area < self.min_area or bw < 2 or bh < 2:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cls_to_idx[name.strip()])
            areas.append(area)
            iscrowd.append(0)

        return boxes, labels, areas, iscrowd

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, f"{name}.jpg")
        xml_path = os.path.join(self.images_dir, f"{name}.xml")
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        if os.path.isfile(xml_path):
            boxes, labels, areas, iscrowd = self._parse_xml(xml_path, w, h)
        else:
            boxes, labels, areas, iscrowd = [], [], [], []

        if len(boxes) == 0:
            # skip empty annotation samples to keep training stable
            return self.__getitem__((idx + 1) % len(self))

        if self.transforms:
            transformed = self.transforms(image=np.array(image), bboxes=boxes, labels=labels)
            img_tensor = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            from torchvision import transforms as T
            img_tensor = T.ToTensor()(image)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.uint8)
        }

        return img_tensor, target


# -----------------------------
# Model utilities
# -----------------------------

def build_ssd(num_classes: int, freeze_backbone: bool = False, image_size: int = 300) -> torch.nn.Module:
    # Download model when calling for the first time
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT, weights_backbone=VGG16_Weights.DEFAULT)
    in_channels = model_utils.retrieve_out_channels(model.backbone, (image_size, image_size))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    return model

# -----------------------------
# Training / Validation loops
# -----------------------------

def detection_collate(batch: List[Tuple[torch.Tensor, Dict]]):
    images, targets = list(zip(*batch))
    # Images already resized to same size, but detection model expects list[Tensor]
    return list(images), list(targets)

def train_one_epoch(model, dataloader, optimizer, device, epoch: int):
    model.train()
    total, count = 0.0, 0

    pbar = tqdm(dataloader, desc=f"Train {epoch}", unit='batch')
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()

        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if torch.isfinite(losses):
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += losses.item()
                count += 1
                pbar.set_postfix(loss=f"{losses.item():.4f}", avg=f"{(total/max(count,1)):.4f}")
        except Exception as e:
            print(f"Batch error: {e}")
            continue

    return total / max(count, 1)


# ===== Metrics Helper Functions (dataset agnostic) =====

def _iou_matrix_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]

    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])

    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)

    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:,None] + area_b[None,:] - inter

    return inter / np.clip(union, 1e-8, None)

def _integrate_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]

    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

def _compute_coco_ap(gt_boxes, preds_per_class):
    """
    gt_boxes: dict[(img_id, cls)] -> list[np.array(4)]
    preds_per_class: dict[cls] -> list{score, img_id, box}
    Returns: overall_map, map_per_iou(dict), ap50, ap75
    """
    iou_thresholds = [round(x/100, 2) for x in range(50, 100, 5)]
    class_ids = sorted({cls for (_im, cls) in gt_boxes.keys()})
    gt_count_per_class = {cls: 0 for cls in class_ids}
    for (img_id, cls), lst in gt_boxes.items():
        gt_count_per_class[cls] += len(lst)

    ap_per_iou = {thr: [] for thr in iou_thresholds}

    for thr in iou_thresholds:
        for cls in class_ids:
            n_gt = gt_count_per_class[cls]
            if n_gt == 0:
                continue

            preds = preds_per_class.get(cls, [])
            preds_sorted = sorted(preds, key=lambda d: d["score"], reverse=True)

            # fresh matched flags per (img, cls)
            matched_flags = {}
            for (img_id, c), lst in gt_boxes.items():
                if c == cls:
                    matched_flags[(img_id, c)] = [False]*len(lst)

            tp, fp = [], []
            for pred in preds_sorted:
                img_id = pred["img_id"]
                key = (img_id, cls)
                matched = False
                if key in gt_boxes:
                    g = np.vstack(gt_boxes[key])
                    ious = _iou_matrix_np(pred["box"][None,:], g)[0]
                    best = np.argmax(ious) if ious.size else -1
                    if best >= 0 and ious[best] >= thr and not matched_flags[key][best]:
                        matched_flags[key][best] = True
                        matched = True
                tp.append(1 if matched else 0)
                fp.append(0 if matched else 1)

            if tp:
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recalls = tp_cum / max(1, n_gt)
                precisions = tp_cum / np.maximum(1, tp_cum + fp_cum)
                ap = _integrate_pr(recalls, precisions)
            else:
                ap = 0.0
            ap_per_iou[thr].append(ap)

    map_per_iou = {thr: (float(np.mean(v)) if v else 0.0) for thr, v in ap_per_iou.items()}
    overall_map = float(np.mean(list(map_per_iou.values()))) if map_per_iou else 0.0
    ap50 = map_per_iou.get(0.5, 0.0)
    ap75 = map_per_iou.get(0.75, 0.0)

    return overall_map, map_per_iou, ap50, ap75, gt_count_per_class, class_ids

def _compute_micro_auprc(gt_boxes, preds_per_class, gt_count_per_class, iou_thr=0.5):
    """
    Micro-averaged AUPRC (area under precision-recall curve) across all classes at a single IoU
    threshold.
    """
    total_gt = sum(gt_count_per_class.values())
    if total_gt == 0:
        return 0.0

    # Flatten predictions
    flat = []
    for cls, plist in preds_per_class.items():
        for p in plist:
            flat.append((p["score"], cls, p["img_id"], p["box"]))
    flat.sort(key=lambda x: x[0], reverse=True)

    matched_flags = {}
    for k, lst in gt_boxes.items():
        matched_flags[k] = [False]*len(lst)

    tp_run = 0
    fp_run = 0
    precisions = []
    recalls = []
    for sc, cls, img_id, box in flat:
        key = (img_id, cls)
        matched = False
        if key in gt_boxes:
            g = np.vstack(gt_boxes[key])
            ious = _iou_matrix_np(box[None,:], g)[0]
            best = np.argmax(ious) if ious.size else -1
            if best >= 0 and ious[best] >= iou_thr and not matched_flags[key][best]:
                matched_flags[key][best] = True
                matched = True
        if matched:
            tp_run += 1
        else:
            fp_run += 1
        precisions.append(tp_run / max(1, tp_run + fp_run))
        recalls.append(tp_run / total_gt)

    if not recalls:
        return 0.0
    r = np.array(recalls); p = np.array(precisions)
    order = np.argsort(r)
    r = r[order]; p = p[order]

    for i in range(p.size - 1, 0, -1):
        p[i-1] = max(p[i-1], p[i])

    return float(np.trapz(p, r))

def _forward_and_collect(model, images, targets, device, gt_boxes, preds_per_class):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # loss (train mode)
    loss_dict = model(images, targets)
    losses = sum(v for v in loss_dict.values())

    # predictions (eval mode)
    model.eval()
    det_outs = model(images)
    model.train()

    for det, tgt in zip(det_outs, targets):
        img_id = int(tgt['image_id'].item()) if tgt['image_id'].numel() == 1 else int(tgt['image_id'][0].item())

        # GT
        gboxes = tgt['boxes'].detach().cpu().numpy()
        glabels = tgt['labels'].detach().cpu().numpy()
        for gbox, glab in zip(gboxes, glabels):
            if glab == 0:
                continue
            gt_boxes[(img_id, int(glab))].append(gbox.astype(np.float32))

        # Preds
        pboxes = det['boxes'].detach().cpu().numpy()
        pscores = det['scores'].detach().cpu().numpy()
        plabels = det['labels'].detach().cpu().numpy()
        for box, sc, lab in zip(pboxes, pscores, plabels):
            if lab == 0:
                continue
            preds_per_class[int(lab)].append({
                "score": float(sc),
                "img_id": img_id,
                "box": box.astype(np.float32)
            })

    return losses

def validate(model, dataloader, device, epoch : int | None = None):
    # Need model in training mode for it to return loss dict (torchvision detection API)
    was_training = model.training
    model.train()
    total_loss = 0.0
    count = 0

    gt_boxes = defaultdict(list)       # (img_id, cls) -> list[box]
    preds_per_class = defaultdict(list)
    
    if epoch is not None:
        pbar = tqdm(dataloader, desc=f"Val   {epoch}", unit="batch")
    else:
        pbar = tqdm(dataloader, desc="Val", unit="batch")

    with torch.no_grad():
        for images, targets in pbar:
            if any(t['boxes'].numel() == 0 for t in targets):
                continue
            try:
                losses = _forward_and_collect(model, images, targets, device, gt_boxes, preds_per_class)
                if torch.isfinite(losses):
                    total_loss += losses.item()
                    count += 1
                    pbar.set_postfix(val_loss=f"{losses.item():.4f}", avg=f"{(total_loss/max(count,1)):.4f}")
            except Exception as e:
                pbar.set_postfix(error=str(e))
                continue

    # Metrics
    overall_map, map_per_iou, ap50, ap75, gt_count_per_class, class_ids = _compute_coco_ap(gt_boxes, preds_per_class)
    auprc = _compute_micro_auprc(gt_boxes, preds_per_class, gt_count_per_class, iou_thr=0.5)

    if epoch is not None:
        print(f"[Epoch {epoch}] COCO mAP(0.50:0.95) {overall_map:.4f} | AP50 {ap50:.4f} | AP75 {ap75:.4f} | Micro AUPRC@0.5 {auprc:.4f}")
    else:
        print(f"COCO mAP(0.50:0.95) {overall_map:.4f} | AP50 {ap50:.4f} | AP75 {ap75:.4f} | Micro AUPRC@0.5 {auprc:.4f}")

    if not was_training:
        model.eval()

    #return total_loss / max(count, 1)
    return total_loss / max(count, 1), gt_boxes, preds_per_class


# -----------------------------
# Prediction & Visualization
# -----------------------------

def plot_curves(train_losses: List[float], val_losses: List[float], out_path: str):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves'); plt.grid(True); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss (log)'); plt.title('Loss (Log)'); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved curves to {out_path}")

# ----- Clean predict_image & visualize -----
def predict_image(model,
                  image_path: str,
                  device: str,
                  class_names: List[str],
                  transform=None,
                  conf_thresh: float = 0.5):
    model.eval()
    pil_image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_image.size

    if transform is not None:
        transformed = transform(image=np.array(pil_image), bboxes=[], labels=[])
        img_tensor = transformed['image']
    else:
        basic = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        img_tensor = basic(image=np.array(pil_image))['image']

    resized_h, resized_w = img_tensor.shape[1], img_tensor.shape[2]
    with torch.no_grad():
        out = model([img_tensor.to(device)])[0]

    scores = out['scores'].cpu().numpy()
    boxes = out['boxes'].cpu().numpy()
    labels = out['labels'].cpu().numpy()
    keep = scores >= conf_thresh
    scores, boxes, labels = scores[keep], boxes[keep], labels[keep]

    if (orig_w, orig_h) != (resized_w, resized_h):
        sx = orig_w / resized_w
        sy = orig_h / resized_h
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

    return {
        'scores': scores,
        'boxes': boxes,
        'labels': labels,
        'orig_size': (orig_w, orig_h),
        'proc_size': (resized_w, resized_h)
    }


def visualize_predictions(image_path: str, predictions: Dict, class_names: List[str], save_path: str | None = None):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    colors = ['red','blue','green','orange','purple','yellow','cyan','magenta','lime','pink']

    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        x1, y1, x2, y2 = box
        cname = class_names[label - 1] if 0 < label <= len(class_names) else f"cls_{label}"
        color = colors[label % len(colors)]
        # outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{cname}:{score:.2f}"
        # Proper bbox for text
        tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), text, font=font)
        # Shift label above box if space, else draw inside
        label_bottom = y1
        label_top = label_bottom - (ty2 - ty1)
        if label_top < 0:
            label_top = y1
            label_bottom = y1 + (ty2 - ty1)
        draw.rectangle([tx1, label_top, tx2, label_bottom], fill=color)
        draw.text((tx1, label_top), text, fill='white', font=font)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        print(f"Saved prediction image -> {save_path}")
    return image


# -----------------------------
# Wrapper combining training & validation with curve saving
# -----------------------------

def train_and_evaluate(model: torch.nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler | None,
                       device: str,
                       epochs: int,
                       output_dir: str,
                       prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    train_losses: List[float] = []
    val_losses: List[float] = []
    
    best_val_loss = float('inf')
    patience = 5  # number of epochs without improvment (otherwise it stops)
    best_epoch = 0
    trigger_times = 0

    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device, epoch + 1)
        #va = validate(model, val_loader, device, epoch + 1)
        va_loss, _, _ = validate(model, val_loader, device, epoch + 1)
        va = va_loss 

        if scheduler:
            scheduler.step()
        train_losses.append(tr)
        val_losses.append(va)

        # Add of Early Stopping to avoid overfitting
        if va < best_val_loss:
            best_val_loss = va
            best_epoch = epoch + 1
            trigger_times = 0
            # Save of the best model
            torch.save({'model_state_dict': model.state_dict()},os.path.join(output_dir, f'{prefix}_best.pth'))
            print(f" New best model at epoch {epoch+1} (val_loss={va:.4f}) saved.")
        else:
            trigger_times += 1
            print(f" No improvement for {trigger_times} epoch(s).")

        if trigger_times >= patience:
            print(f" Early stopping at epoch {epoch+1}. Best val_loss = {best_val_loss:.4f}")
            break
        
        # intermediate checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(output_dir, f'{prefix}_epoch_{epoch+1}.pth'))

    if len(train_losses) > 0:
        plot_curves(train_losses, val_losses, os.path.join(output_dir, f'{prefix}_curves.png'))
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(output_dir, f'{prefix}_final.pth'))
        print(f"\n Training completed after {epoch+1} epochs. Best val_loss={best_val_loss:.4f} (epoch {best_epoch})")

    return train_losses, val_losses


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ======================================================
    # COCO DATASET
    # ======================================================
    print("\n=== Loading COCO subset ===")

    # Paths (subset versions for faster experimentation)
    train_images_dir = 'subset_coco/subset_train_images_2014'
    val_images_dir   = 'subset_coco/subset_val_images_2014'
    test_images_dir  = 'subset_coco/test_images_2014'
    train_ann        = 'subset_coco/train_val_annotations/subset_instances_train2014.json'
    val_ann          = 'subset_coco/train_val_annotations/subset_instances_val2014.json'
    saved_model_path = 'outputs/ssd300_coco_final.pth'
    test_image_name  = 'COCO_test2014_000000000001.jpg'

    # Inline transforms (train / val)
    image_size = 300

    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.0))

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.0))

    # Datasets / loaders
    coco_train = COCODataset(train_images_dir, train_ann, transforms=train_transform, min_area=32*32)
    coco_val   = COCODataset(val_images_dir,   val_ann,   transforms=val_transform,   min_area=32*32)
    num_classes_coco = len(coco_train.class_names) + 1  # + background
    print(f"COCO classes ({num_classes_coco-1}): {coco_train.class_names}")

    train_loader = DataLoader(coco_train, batch_size=8, shuffle=True,  collate_fn=detection_collate, num_workers=0)
    val_loader   = DataLoader(coco_val,   batch_size=8, shuffle=False, collate_fn=detection_collate, num_workers=0)

    # Model + optim
    model = build_ssd(num_classes_coco, freeze_backbone=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    # Training epochs
    epochs_coco = 1

    if epochs_coco > 0:
        train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, epochs_coco, 'outputs', 'ssd300_coco')

    # prediction on a test image
    test_image_path = test_images_dir + test_image_name
    if os.path.exists(test_image_path):
        print("COCO prediction test...")
        preds = predict_image(model, test_image_path, device, coco_train.class_names,
                              transform=val_transform, conf_thresh=0.7)
        visualize_predictions(test_image_path, preds, coco_train.class_names, save_path='outputs/prediction_coco.jpg')
        print(f"Detections: {len(preds['labels'])}")
        for l, s in zip(preds['labels'], preds['scores']):
            cname = coco_train.class_names[l - 1] if l > 0 else 'bg'
            print(f"  {cname}: {s:.3f}")

    # inference using a checkpoint
    model_checkpoint_path = saved_model_path
    model = build_ssd(num_classes_coco, freeze_backbone=True).to(device)
    if os.path.isfile(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device)['model_state_dict'])
        print(f"Loaded checkpoint from {model_checkpoint_path}")
    
    #val_loss = validate(model, val_loader, device)
    val_loss, gt_boxes, preds = validate(model, val_loader, device)



    # ======================================================
    # USURE DATASET
    # ======================================================
    print("\n=== Loading Usure dataset ===")

    usure_train_list  = 'data_tu/image_names_train_new.txt'
    usure_val_list    = 'data_tu/image_names_val_new.txt'
    usure_test_list   = 'data_tu/image_names_test_new.txt'
    usure_images_dir  = 'data_tu/image_base'

    # Load class filter (optional) from parameters
    with open('utilities/parameters.json', 'r', encoding='utf-8') as f:
        params = json.load(f)
    class_filter = params.get('class_filter', None)

    # Datasets / loaders
    usure_train = UsureDataset(usure_train_list, usure_images_dir, transforms=train_transform, class_names=class_filter, min_area=32*32)
    usure_val   = UsureDataset(usure_val_list,   usure_images_dir, transforms=val_transform,   class_names=class_filter, min_area=32*32)
    usure_test  = UsureDataset(usure_test_list,  usure_images_dir, transforms=val_transform,   class_names=class_filter,min_area=32*32)

    num_usure_classes = len(usure_train.class_names) + 1
    print(f"Usure classes ({num_usure_classes-1}): {usure_train.class_names}")

    usure_train_loader = DataLoader(usure_train, batch_size=8, shuffle=True,  collate_fn=detection_collate, num_workers=0)
    usure_val_loader   = DataLoader(usure_val,   batch_size=8, shuffle=False, collate_fn=detection_collate, num_workers=0)
    usure_test_loader  = DataLoader(usure_test,  batch_size=8, shuffle=False, collate_fn=detection_collate, num_workers=0)


    # Build a fresh model for Usure
    model_usure = build_ssd(num_usure_classes, freeze_backbone=False).to(device)

    optimizer_u = torch.optim.AdamW(model_usure.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_u = torch.optim.lr_scheduler.MultiStepLR(optimizer_u, milestones=[10, 15], gamma=0.1)

    # Short demo run; extend epochs for real training
    epochs_usure = 30
    train_and_evaluate(model_usure, usure_train_loader, usure_val_loader, optimizer_u, scheduler_u, device, epochs_usure, 'outputs', 'ssd300_usure')

    # Single prediction on first validation sample
    with open(usure_val_list, 'r', encoding='utf-8') as f:
        first_name = f.readline().strip()
    usure_test_image = os.path.join(usure_images_dir, f"{first_name}.jpg")
    if os.path.exists(usure_test_image):
        print("Usure prediction test...")
        preds_u = predict_image(model_usure, usure_test_image, device, usure_train.class_names,
                                transform=val_transform, conf_thresh=0.2)
        visualize_predictions(usure_test_image, preds_u, usure_train.class_names, save_path='outputs/prediction_usure.jpg')
        print(f"Usure detections: {len(preds_u['labels'])}")
        for l, s in zip(preds_u['labels'], preds_u['scores']):
            cname = usure_train.class_names[l - 1] if l > 0 else 'bg'
            print(f"  {cname}: {s:.3f}")
            
    print("\n=== Evaluation finale sur TEST SET ===")

    test_loss, gt_boxes, preds_per_class = validate(model_usure, usure_test_loader, device)
    print(f"Test loss = {test_loss:.4f}")
    
    os.makedirs("outputs", exist_ok=True)
    
    with open("outputs/gt_boxes.pkl", "wb") as f:
        pickle.dump(gt_boxes, f)
    
    with open("outputs/preds_per_class.pkl", "wb") as f:
        pickle.dump(preds_per_class, f)
    
    with open("outputs/class_names.json", "w") as f:
        json.dump(usure_train.class_names, f)


if __name__ == '__main__':
    main()
