# ============================================================
# training.py
# Entraînement FasterRCNN + mAP + AUPRC
# ============================================================

import os
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TemoinVOCDataset, CLASS_NAMES
from model import get_fasterrcnn_resnet50_fpn
from metrics import compute_coco_ap, compute_micro_auprc


# ------------------------------------------------------------
# Collate FN
# ------------------------------------------------------------

def detection_collate(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


# ------------------------------------------------------------
# Entraînement d'une époque
# ------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total, count = 0.0, 0
    pbar = tqdm(dataloader, desc=f"Train {epoch}", unit="batch")

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        try:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total += loss.item()
                count += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 avg=f"{(total / count):.4f}")
        except Exception as e:
            print(f"[Batch Error] {e}")
            continue

    return total / max(count, 1)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate(model, dataloader, device, epoch=None):
    model.train()  # pour certaines versions
    total_loss, count = 0.0, 0

    gt_boxes = defaultdict(list)
    preds_per_class = defaultdict(list)

    pbar = tqdm(dataloader, desc=f"Val {epoch}", unit="batch")

    with torch.no_grad():
        for images, targets in pbar:

            if any(t["boxes"].numel() == 0 for t in targets):
                continue

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Loss
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())

            if torch.isfinite(loss):
                total_loss += loss.item()
                count += 1

            # Prédictions
            detections = model(images)

            for det, tgt in zip(detections, targets):
                img_id = int(tgt["image_id"].item())

                # GT
                for gbox, glab in zip(
                        tgt["boxes"].cpu().numpy(),
                        tgt["labels"].cpu().numpy()):
                    gt_boxes[(img_id, int(glab))].append(gbox)

                # Predictions
                for pbox, score, lab in zip(
                        det["boxes"].cpu().numpy(),
                        det["scores"].cpu().numpy(),
                        det["labels"].cpu().numpy()):
                    preds_per_class[int(lab)].append({
                        "score": float(score),
                        "img_id": img_id,
                        "box": pbox
                    })

    map_val, ap50, ap75 = compute_coco_ap(gt_boxes, preds_per_class)
    auprc_val = compute_micro_auprc(gt_boxes, preds_per_class, defaultdict(int))

    if epoch:
        print(f"[Epoch {epoch}] mAP={map_val:.4f} | AP50={ap50:.4f} | AUPRC={auprc_val:.4f}")

    return total_loss / max(count, 1), map_val, ap50, ap75, auprc_val


# ------------------------------------------------------------
# Entraînement complet
# ------------------------------------------------------------

def train_and_evaluate(model, train_loader, val_loader,
                       optimizer, scheduler, device,
                       epochs, output_dir, prefix):

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, f"runs_{prefix}"))

    metrics = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "mAP": [], "AP50": [], "AP75": [], "AUPRC": []
    }

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        va_loss, map_val, ap50_val, ap75_val, auprc_val = validate(
            model, val_loader, device, epoch
        )

        if scheduler:
            scheduler.step()

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(tr_loss)
        metrics["val_loss"].append(va_loss)
        metrics["mAP"].append(map_val)
        metrics["AP50"].append(ap50_val)
        metrics["AP75"].append(ap75_val)
        metrics["AUPRC"].append(auprc_val)

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("mAP/val", map_val, epoch)

        if epoch % 5 == 0 or epoch == epochs:
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(output_dir, f"{prefix}_epoch_{epoch}.pth")
            )

    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, f"{prefix}_metrics.csv"))
    print("Training completed.")


# ------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------

def main():

    train_img = "data/train/images"
    train_xml = "data/train/annotations"

    val_img = "data/val/images"
    val_xml = "data/val/annotations"

    train_dataset = TemoinVOCDataset(train_img, train_xml)
    val_dataset   = TemoinVOCDataset(val_img, val_xml)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=detection_collate)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=4, collate_fn=detection_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 1 + len(CLASS_NAMES)  # background + classes
    model = get_fasterrcnn_resnet50_fpn(num_classes=num_classes, freeze_backbone=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_and_evaluate(
        model, train_loader, val_loader,
        optimizer, scheduler, device,
        epochs=20,
        output_dir="outputs_temoin",
        prefix="fasterrcnn_temoin",
    )


if __name__ == "__main__":
    main()
