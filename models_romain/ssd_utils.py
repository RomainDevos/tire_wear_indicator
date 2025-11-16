import os
import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.detection import _utils as model_utils
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# -----------------------------
# Build SSD model
# -----------------------------

def build_ssd(num_classes: int, freeze_backbone: bool = False, image_size: int = 300) -> torch.nn.Module:
    """
    Construit un modèle SSD300 avec VGG16 backbone.
    Peut être initialisé avec ou sans gel du backbone.
    """
    model = ssd300_vgg16(weights=None, weights_backbone=VGG16_Weights.DEFAULT)
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
        print(" Backbone frozen (training only detection head).")
    else:
        print(" Backbone trainable.")

    return model


# -----------------------------
# Prediction utilities
# -----------------------------

def predict_image(model, image_path: str, device: str, class_names, transform=None, conf_thresh: float = 0.5):
    """
    Prédit les objets sur une image avec le modèle SSD.
    Retourne dictionnaire avec boxes, scores, labels.
    """
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

    with torch.no_grad():
        out = model([img_tensor.to(device)])[0]

    scores = out['scores'].cpu().numpy()
    boxes = out['boxes'].cpu().numpy()
    labels = out['labels'].cpu().numpy()

    keep = scores >= conf_thresh
    scores, boxes, labels = scores[keep], boxes[keep], labels[keep]

    return {'scores': scores, 'boxes': boxes, 'labels': labels}


# -----------------------------
# Visualization
# -----------------------------

def visualize_predictions(image_path: str, predictions, class_names, save_path: str | None = None):
    """
    Affiche et/ou sauvegarde l'image avec les prédictions dessinées.
    """
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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{cname}:{score:.2f}"
        tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), text, font=font)
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
        print(f" Saved prediction image -> {save_path}")

    return image
 
