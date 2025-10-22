
import numpy as np
from collections import defaultdict

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU between all boxes in a and b."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.clip(union, 1e-8, None)


def integrate_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Integration under precision-recall curve."""
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_coco_ap(gt_boxes, preds_per_class):
    """Compute COCO-style mAP, AP50, AP75, AUPRC."""
    iou_thresholds = [round(x / 100, 2) for x in range(50, 100, 5)]
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
            matched_flags = {(img_id, c): [False] * len(lst)
                             for (img_id, c), lst in gt_boxes.items() if c == cls}
            tp, fp = [], []
            for pred in preds_sorted:
                img_id = pred["img_id"]
                key = (img_id, cls)
                matched = False
                if key in gt_boxes:
                    g = np.vstack(gt_boxes[key])
                    ious = iou_matrix(pred["box"][None, :], g)[0]
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
                ap = integrate_pr(recalls, precisions)
            else:
                ap = 0.0
            ap_per_iou[thr].append(ap)

    map_per_iou = {thr: (float(np.mean(v)) if v else 0.0) for thr, v in ap_per_iou.items()}
    overall_map = float(np.mean(list(map_per_iou.values()))) if map_per_iou else 0.0
    ap50 = map_per_iou.get(0.5, 0.0)
    ap75 = map_per_iou.get(0.75, 0.0)

    return overall_map, ap50, ap75


def compute_micro_auprc(gt_boxes, preds_per_class, gt_count_per_class, iou_thr=0.5):
    """Compute micro-averaged AUPRC across all classes."""
    total_gt = sum(gt_count_per_class.values())
    if total_gt == 0:
        return 0.0

    flat = []
    for cls, plist in preds_per_class.items():
        for p in plist:
            flat.append((p["score"], cls, p["img_id"], p["box"]))
    flat.sort(key=lambda x: x[0], reverse=True)

    matched_flags = {k: [False] * len(lst) for k, lst in gt_boxes.items()}
    tp_run, fp_run = 0, 0
    precisions, recalls = [], []
    for sc, cls, img_id, box in flat:
        key = (img_id, cls)
        matched = False
        if key in gt_boxes:
            g = np.vstack(gt_boxes[key])
            ious = iou_matrix(box[None, :], g)[0]
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

    r = np.array(recalls)
    p = np.array(precisions)
    order = np.argsort(r)
    r, p = r[order], p[order]
    for i in range(p.size - 1, 0, -1):
        p[i - 1] = max(p[i - 1], p[i])

    return float(np.trapz(p, r))
