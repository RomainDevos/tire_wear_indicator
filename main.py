import os
import json
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === Import de nos modules internes ===
from datasets.coco_dataset import COCODataset
from datasets.usure_dataset import UsureDataset
from models.ssd_utils import build_ssd
from training.train_eval import train_and_evaluate, detection_collate, validate
from training.metrics import compute_coco_ap

from utilities.visualize import predict_image, visualize_predictions  # si tu veux isoler les visualisations

# -------------------------------------------------------------
# Configuration principale
# -------------------------------------------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Using device: {device}")

    # ======================================================
    # ===   COCO SUBSET (pré-entraînement, facultatif)   ===
    # ======================================================
    print("\n=== Loading COCO subset ===")

    train_images_dir = 'subset_coco/subset_train_images_2014'
    val_images_dir   = 'subset_coco/subset_val_images_2014'
    test_images_dir  = 'subset_coco/test_images_2014'
    train_ann        = 'subset_coco/train_val_annotations/subset_instances_train2014.json'
    val_ann          = 'subset_coco/train_val_annotations/subset_instances_val2014.json'
    saved_model_path = 'outputs/ssd300_coco_final.pth'

    image_size = 300

    # Transforms
    transform_train = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    transform_val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    coco_train = COCODataset(train_images_dir, train_ann, transforms=transform_train, min_area=32*32)
    coco_val   = COCODataset(val_images_dir,   val_ann,   transforms=transform_val,   min_area=32*32)
    num_classes_coco = len(coco_train.class_names) + 1

    print(f"COCO classes ({num_classes_coco-1}): {coco_train.class_names[:5]} ...")

    train_loader = DataLoader(coco_train, batch_size=8, shuffle=True,  collate_fn=detection_collate)
    val_loader   = DataLoader(coco_val,   batch_size=8, shuffle=False, collate_fn=detection_collate)

    # === Modèle COCO (optionnel) ===
    model_coco = build_ssd(num_classes_coco, freeze_backbone=True).to(device)
    optimizer_coco = torch.optim.AdamW(model_coco.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_coco = torch.optim.lr_scheduler.MultiStepLR(optimizer_coco, milestones=[10, 15], gamma=0.1)

    epochs_coco = 0  # Mets >0 pour entraîner sur COCO
    if epochs_coco > 0:
        train_and_evaluate(model_coco, train_loader, val_loader,
                           optimizer_coco, scheduler_coco, device, epochs_coco,
                           'outputs', 'ssd300_coco')
    else:
        print("⏩ Skipping COCO training (using pretrained backbone only)")

    # ======================================================
    # ===   USURE DATASET (fine-tuning sur ton dataset)  ===
    # ======================================================
    print("\n=== Loading Usure dataset ===")

    usure_train_list  = '../data_tu/data_tu/image_names_train.txt'
    usure_val_list    = '../data_tu/data_tu/image_names_val.txt'
    usure_images_dir  = '../data_tu/data_tu/image_base'

    with open('utilities/parameters.json', 'r', encoding='utf-8') as f:
        params = json.load(f)
    class_filter = params.get('class_filter', None)

    usure_train = UsureDataset(usure_train_list, usure_images_dir,
                               transforms=transform_train,
                               class_names=class_filter, min_area=32*32)
    usure_val = UsureDataset(usure_val_list, usure_images_dir,
                             transforms=transform_val,
                             class_names=class_filter, min_area=32*32)
    num_usure_classes = len(usure_train.class_names) + 1
    print(f"Usure classes ({num_usure_classes-1}): {usure_train.class_names}")

    train_loader_u = DataLoader(usure_train, batch_size=8, shuffle=True, collate_fn=detection_collate)
    val_loader_u   = DataLoader(usure_val,   batch_size=8, shuffle=False, collate_fn=detection_collate)

    # === Charger backbone COCO et geler ===
    model_usure = build_ssd(num_usure_classes, freeze_backbone=False).to(device)
    coco_weights_path = 'C:/Users/romai/.cache/torch/hub/checkpoints/ssd300_vgg16_coco-b556d3b4.pth'
    coco_weights = torch.load(coco_weights_path, map_location=device)
    backbone_dict = {k: v for k, v in coco_weights.items() if k.startswith('backbone.')}
    missing, unexpected = model_usure.load_state_dict(backbone_dict, strict=False)
    print(f"Backbone weights loaded from COCO ({len(backbone_dict)} tensors)")
    print(f"ℹ Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    for p in model_usure.backbone.parameters():
        p.requires_grad = False
    print(" Backbone frozen (training detection head only)")

    optimizer_u = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_usure.parameters()),
                                    lr=1e-4, weight_decay=1e-4)
    scheduler_u = torch.optim.lr_scheduler.MultiStepLR(optimizer_u, milestones=[10, 15], gamma=0.1)

    epochs_usure = 5
    train_and_evaluate(model_usure, train_loader_u, val_loader_u,
                       optimizer_u, scheduler_u, device, epochs_usure,
                       'outputs', 'ssd300_usure_frozen')

    # === Test sur une image ===
    with open(usure_val_list, 'r', encoding='utf-8') as f:
        test_name = f.readline().strip()
    test_img_path = os.path.join(usure_images_dir, f"{test_name}.jpg")

    if os.path.exists(test_img_path):
        print("🔍 Running prediction on sample image...")
        preds = predict_image(model_usure, test_img_path, device, usure_train.class_names,
                              transform=transform_val, conf_thresh=0.3)
        visualize_predictions(test_img_path, preds, usure_train.class_names,
                              save_path='outputs/prediction_usure_frozen.jpg')
        print(f" Detections ({len(preds['labels'])}):")
        for l, s in zip(preds['labels'], preds['scores']):
            cname = usure_train.class_names[l - 1] if l > 0 else 'bg'
            print(f"  {cname}: {s:.3f}")

if __name__ == "__main__":
    main()
