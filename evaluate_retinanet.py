import os
import torch
import torchvision
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as T

# -----------------------
# IoU
# -----------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# -----------------------
# Model
# -----------------------
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
model = retinanet_resnet50_fpn(weights=weights)

num_classes = 2
in_channels = model.head.classification_head.conv[0].out_channels
num_anchors = model.head.classification_head.num_anchors

model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
    in_channels, num_anchors, num_classes
)

model.load_state_dict(torch.load("retinanet_ucas.pth"))
model.eval()
model.to(device)

# -----------------------
# Paths
# -----------------------
IMG_DIR = "UCAS-AOD-VOC/images/test"
ANN_DIR = "UCAS-AOD-VOC/annotations/test"

transform = T.Compose([
    T.Resize((600, 600)),
    T.ToTensor()
])

TP = FP = FN = 0
iou_thresh = 0.5

# -----------------------
# Evaluation
# -----------------------
for img_name in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_name)
    ann_path = os.path.join(ANN_DIR, img_name.replace(".png", ".xml"))

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((600, 600))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    preds = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    preds = preds[scores > 0.5]

    gt_boxes = []
    tree = ET.parse(ann_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text) * 600 / orig_w
        ymin = float(b.find("ymin").text) * 600 / orig_h
        xmax = float(b.find("xmax").text) * 600 / orig_w
        ymax = float(b.find("ymax").text) * 600 / orig_h
        gt_boxes.append([xmin, ymin, xmax, ymax])

    matched = set()
    for p in preds:
        found = False
        for i, g in enumerate(gt_boxes):
            if i in matched:
                continue
            if compute_iou(p, g) >= iou_thresh:
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
map50 = precision * recall

print("===== RetinaNet Evaluation =====")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"mAP@0.5:   {map50:.3f}")
