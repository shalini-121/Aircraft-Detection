import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as T


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

            xmin = xmin * 600 / orig_w
            xmax = xmax * 600 / orig_w
            ymin = ymin * 600 / orig_h
            ymax = ymax * 600 / orig_h

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return self.transform(img), target

    def __len__(self):
        return len(self.imgs)
