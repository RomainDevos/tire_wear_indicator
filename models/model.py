# ============================================================
# model.py
# FasterRCNN_ResNet50_FPN avec tête personnalisée
# ============================================================

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_fasterrcnn_resnet50_fpn(num_classes, freeze_backbone=True):
    """
    num_classes = 1 + nombre de classes (background + classes réelles)
    Ex : 1 + 7 = 8
    """
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="COCO_V1"
        )
    except:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Remplacement de la tête
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    return model
