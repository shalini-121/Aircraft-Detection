import os
import random
import shutil

BASE_DIR = "UCAS-AOD-VOC"
IMG_DIR = os.path.join(BASE_DIR, "images")
ANN_DIR = os.path.join(BASE_DIR, "annotations")

# Create split folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(IMG_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(ANN_DIR, split), exist_ok=True)

# Get all images
images = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
images.sort()          # keep order stable
random.seed(42)        # reproducibility
random.shuffle(images)

# Split counts (1000 images)
train_imgs = images[:700]
val_imgs = images[700:900]
test_imgs = images[900:]

def move_files(img_list, split):
    for img in img_list:
        xml = img.replace(".png", ".xml")

        shutil.move(
            os.path.join(IMG_DIR, img),
            os.path.join(IMG_DIR, split, img)
        )
        shutil.move(
            os.path.join(ANN_DIR, xml),
            os.path.join(ANN_DIR, split, xml)
        )

move_files(train_imgs, "train")
move_files(val_imgs, "val")
move_files(test_imgs, "test")

print("✅ Dataset successfully split into train / val / test")
