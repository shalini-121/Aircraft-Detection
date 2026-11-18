import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from PIL import Image

# ======================
# Dataset
# ======================
class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.imgs = sorted(os.listdir(os.path.join(root, "images", split)))
        self.transform = T.ToTensor()

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, "images", self.split, img_name)
        ann_path = os.path.join(
            self.root, "annotations", self.split,
            img_name.replace(".png", ".xml")
        )

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize ONCE
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

            if xmax <= xmin or ymax <= ymin:
                continue

            xmin = xmin * 600 / orig_w
            xmax = xmax * 600 / orig_w
            ymin = ymin * 600 / orig_h
            ymax = ymax * 600 / orig_h

            xmin = max(0, min(xmin, 599))
            ymin = max(0, min(ymin, 599))
            xmax = max(1, min(xmax, 600))
            ymax = max(1, min(ymax, 600))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return self.transform(img), target

    def __len__(self):
        return len(self.imgs)


# ======================
# Model
# ======================
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
# Freeze backbone to reduce GPU memory
for param in model.backbone.parameters():
    param.requires_grad = False

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

device = torch.device("cuda")
model.to(device)

# ======================
# DataLoader
# ======================
train_dataset = AircraftDataset("UCAS-AOD-VOC", "train")

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# ======================
# Optimizer
# ======================
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# ======================
# Training Loop
# ======================
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()


        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f}")

# ======================
# Save Model
# ======================
torch.save(model.state_dict(), "faster_rcnn_ucas.pth")
print("✅ Training completed and model saved")
