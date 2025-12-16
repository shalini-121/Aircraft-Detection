import os
import torch
import torchvision
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader

# --------------------------------------------------
# Dataset Class (INLINE – no import issues)
# --------------------------------------------------
class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        self.imgs = sorted(os.listdir(self.img_dir))
        self.transform = T.ToTensor()

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(
            self.ann_dir, img_name.replace(".png", ".xml")
        )

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize image
        img = img.resize((600, 600))

        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            ymin = float(b.find("ymin").text)
            xmax = float(b.find("xmax").text)
            ymax = float(b.find("ymax").text)

            # Scale boxes
            xmin = xmin * 600 / orig_w
            xmax = xmax * 600 / orig_w
            ymin = ymin * 600 / orig_h
            ymax = ymax * 600 / orig_h

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        # Skip empty annotations
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # aircraft = 1

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return self.transform(img), target

    def __len__(self):
        return len(self.imgs)


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Dataset & Loader
# --------------------------------------------------
ROOT = "UCAS-AOD-VOC"

train_dataset = AircraftDataset(ROOT, "train")
train_loader = DataLoader(
    train_dataset,
    batch_size=1,      # VERY IMPORTANT for 4GB GPU
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# --------------------------------------------------
# RetinaNet Model (CORRECT WAY)
# --------------------------------------------------
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights

weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
model = retinanet_resnet50_fpn(weights=weights)

num_classes = 2
in_channels = model.head.classification_head.conv[0].out_channels
num_anchors = model.head.classification_head.num_anchors

model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
    in_channels,
    num_anchors,
    num_classes
)

model.to(device)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
num_epochs = 8

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# --------------------------------------------------
# Save Model
# --------------------------------------------------
torch.save(model.state_dict(), "retinanet_ucas.pth")
print("✅ RetinaNet training completed and model saved")
