
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from collections import defaultdict
from .metrics import compute_coco_ap, compute_micro_auprc


def detection_collate(batch):
    """Custom collate_fn for SSD training."""
    images, targets = list(zip(*batch))
    return list(images), list(targets)


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
            losses = sum(loss for loss in loss_dict.values())
            if torch.isfinite(losses):
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += losses.item()
                count += 1
                pbar.set_postfix(loss=f"{losses.item():.4f}", avg=f"{(total / max(count, 1)):.4f}")
        except Exception as e:
            print(f"Batch error: {e}")
            continue
    return total / max(count, 1)


def validate(model, dataloader, device, epoch=None):
    model.train()
    total_loss, count = 0.0, 0
    gt_boxes = defaultdict(list)
    preds_per_class = defaultdict(list)

    pbar = tqdm(dataloader, desc=f"Val {epoch}" if epoch else "Val", unit="batch")
    with torch.no_grad():
        for images, targets in pbar:
            if any(t['boxes'].numel() == 0 for t in targets):
                continue
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(v for v in loss_dict.values())
            if torch.isfinite(losses):
                total_loss += losses.item()
                count += 1
                pbar.set_postfix(val_loss=f"{losses.item():.4f}", avg=f"{(total_loss/max(count,1)):.4f}")
            detections = model(images)
            for det, tgt in zip(detections, targets):
                img_id = int(tgt["image_id"].item())
                for gbox, glab in zip(tgt["boxes"].cpu().numpy(), tgt["labels"].cpu().numpy()):
                    gt_boxes[(img_id, int(glab))].append(gbox)
                for pbox, score, lab in zip(det["boxes"].cpu().numpy(), det["scores"].cpu().numpy(), det["labels"].cpu().numpy()):
                    preds_per_class[int(lab)].append({"score": float(score), "img_id": img_id, "box": pbox})

    map_val, ap50, ap75 = compute_coco_ap(gt_boxes, preds_per_class)
    auprc_val = compute_micro_auprc(gt_boxes, preds_per_class, defaultdict(int))
    if epoch:
        print(f"[Epoch {epoch}] mAP: {map_val:.4f} | AP50: {ap50:.4f} | AUPRC: {auprc_val:.4f}")

    return total_loss / max(count, 1), map_val, ap50, ap75, auprc_val


def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, epochs, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, f"runs_{prefix}"))
    metrics = {"epoch": [], "train_loss": [], "val_loss": [], "mAP": [], "AP50": [], "AP75": [], "AUPRC": []}

    for epoch in range(epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, epoch + 1)
        va_loss, map_val, ap50_val, ap75_val, auprc_val = validate(model, val_loader, device, epoch + 1)
        if scheduler:
            scheduler.step()
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(tr_loss)
        metrics["val_loss"].append(va_loss)
        metrics["mAP"].append(map_val)
        metrics["AP50"].append(ap50_val)
        metrics["AP75"].append(ap75_val)
        metrics["AUPRC"].append(auprc_val)
        writer.add_scalar("Loss/train", tr_loss, epoch + 1)
        writer.add_scalar("Loss/val", va_loss, epoch + 1)
        writer.add_scalar("mAP/val", map_val, epoch + 1)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save({"model_state_dict": model.state_dict()},
                       os.path.join(output_dir, f"{prefix}_epoch_{epoch+1}.pth"))
    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, f"{prefix}_metrics.csv"), index=False)
    writer.close()
    print(f"Training finished — results saved to {output_dir}")
