import os
import cv2
import xml.etree.ElementTree as ET

INPUT_DIR = "PLANE"


OUTPUT_DIR = "UCAS-AOD-VOC"

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)

def create_voc_xml(img_path, boxes, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w, c = img.shape

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = os.path.basename(img_path)

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "aircraft"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))

    tree = ET.ElementTree(annotation)
    tree.write(save_path)

for file in os.listdir(INPUT_DIR):
    if file.lower().endswith(".png"):
        base = file.replace(".png", "")
        txt_path = os.path.join(INPUT_DIR, base + ".txt")
        img_path = os.path.join(INPUT_DIR, file)

        if not os.path.exists(txt_path):
            continue

        boxes = []
        with open(txt_path, "r") as f:
            for line in f.readlines():
                vals = list(map(float, line.strip().split()))
                xs = vals[0:8:2]
                ys = vals[1:8:2]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                boxes.append((xmin, ymin, xmax, ymax))

        create_voc_xml(
            img_path,
            boxes,
            os.path.join(OUTPUT_DIR, "annotations", base + ".xml")
        )

        os.rename(img_path, os.path.join(OUTPUT_DIR, "images", file))

print("✅ UCAS-AOD conversion completed!")
