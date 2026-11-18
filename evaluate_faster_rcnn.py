import os
import torch
import torchvision
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as T

# -----------------------
# IoU Function
# -----------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# -----------------------
# Load Model
# -----------------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("faster_rcnn_ucas.pth", map_location=device))
model.eval()
model.to(device)

transform = T.Compose([
    T.ToTensor()
])

# -----------------------
# Dataset Paths
# -----------------------
IMG_DIR = "UCAS-AOD-VOC/images/test"
ANN_DIR = "UCAS-AOD-VOC/annotations/test"

TP = 0
FP = 0
FN = 0
iou_threshold = 0.5

# -----------------------
# Evaluation Loop
# -----------------------
for img_name in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_name)
    ann_path = os.path.join(ANN_DIR, img_name.replace(".png", ".xml"))

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((600, 600))

    image_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]

    pred_boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    pred_boxes = pred_boxes[scores > 0.5]

    # -----------------------
    # Load GT boxes (scaled)
    # -----------------------
    tree = ET.parse(ann_path)
    root = tree.getroot()

    gt_boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")

        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)

        xmin = xmin * 600 / orig_w
        xmax = xmax * 600 / orig_w
        ymin = ymin * 600 / orig_h
        ymax = ymax * 600 / orig_h

        gt_boxes.append([xmin, ymin, xmax, ymax])

    matched = set()

    # -----------------------
    # Match predictions
    # -----------------------
    for pb in pred_boxes:
        found = False
        for i, gb in enumerate(gt_boxes):
            if i in matched:
                continue
            if compute_iou(pb, gb) >= iou_threshold:
                TP += 1
                matched.add(i)
                found = True
                break
        if not found:
            FP += 1

    FN += len(gt_boxes) - len(matched)

# -----------------------
# Metrics
# -----------------------
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
map_50 = precision * recall  # simplified mAP@0.5

print("===== Faster R-CNN Evaluation =====")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"mAP@0.5:   {map_50:.3f}")
