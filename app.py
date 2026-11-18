import os
import io
import base64
import torch
import torchvision
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw
import torchvision.transforms as T

# ======================
# Flask Setup
# ======================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================
# Load Models
# ======================
models = {}

# ---------- Faster R-CNN ----------
try:
    print("Loading Faster R-CNN...")
    faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2
    in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
    faster_rcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    faster_rcnn.load_state_dict(torch.load("faster_rcnn_ucas.pth", map_location=device))
    faster_rcnn.to(device).eval()
    models["faster_rcnn"] = faster_rcnn
    print("✓ Faster R-CNN loaded successfully")
except Exception as e:
    print("✗ Faster R-CNN failed:", e)
# ---------- RetinaNet (FIXED) ----------
try:
    print("Loading RetinaNet...")

    retinanet = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None,
        num_classes=2   # 🔥 THIS IS THE FIX
    )

    retinanet.load_state_dict(
        torch.load("retinanet_ucas.pth", map_location=device)
    )

    retinanet.to(device).eval()
    models["retinanet"] = retinanet
    print("✓ RetinaNet loaded")

except Exception as e:
    print("✗ RetinaNet failed:", e)

print("Available models:", list(models.keys()))

default_model = "faster_rcnn" if "faster_rcnn" in models else list(models.keys())[0]

# ======================
# Transforms
# ======================
transform = T.Compose([T.ToTensor()])


# ======================
# Utility Functions
# ======================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def non_max_suppression(boxes, scores, iou_thresh=0.4):
    if len(boxes) == 0:
        return np.array([]), np.array([])

    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = scores.argsort()[::-1]

    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)

        if idxs.size == 1:
            break

        xx1 = np.maximum(boxes[i][0], boxes[idxs[1:]][:, 0])
        yy1 = np.maximum(boxes[i][1], boxes[idxs[1:]][:, 1])
        xx2 = np.minimum(boxes[i][2], boxes[idxs[1:]][:, 2])
        yy2 = np.minimum(boxes[i][3], boxes[idxs[1:]][:, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area1 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        area2 = (boxes[idxs[1:]][:, 2] - boxes[idxs[1:]][:, 0]) * \
                (boxes[idxs[1:]][:, 3] - boxes[idxs[1:]][:, 1])

        iou = inter / (area1 + area2 - inter)
        idxs = idxs[1:][iou < iou_thresh]

    return boxes[keep], scores[keep]


def detect(image, model, conf=0.4):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    mask = scores >= conf
    boxes = boxes[mask]
    scores = scores[mask]

    boxes, scores = non_max_suppression(boxes, scores)

    return boxes, scores


def draw_boxes(image, boxes, scores):
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 12), f"Aircraft {score:.2f}", fill="red")
    return image


# ======================
# Routes
# ======================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models")
def get_models():
    available = list(models.keys())
    return jsonify({
        "models": available,
        "available_models": available,
        "default": default_model,
        "default_model": default_model
    })


@app.route("/detect", methods=["POST"])
def detect_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    model_name = request.form.get("model", default_model)
    conf = float(request.form.get("confidence", 0.4))

    print(f"\n{'='*50}")
    print(f"🔍 Detection Request:")
    print(f"   Model: {model_name}")
    print(f"   Confidence Threshold: {conf}")
    print(f"   File: {file.filename}")
    print(f"{'='*50}")

    if model_name not in models:
        print(f"❌ ERROR: Model '{model_name}' not available!")
        print(f"   Available models: {list(models.keys())}")
        return jsonify({"error": f"Model '{model_name}' not available"}), 400

    image = Image.open(file.stream).convert("RGB")
    print(f"📐 Image size: {image.width}x{image.height}")
    
    boxes, scores = detect(image, models[model_name], conf)
    print(f"✅ Detections found: {len(boxes)}")
    if len(boxes) > 0:
        print(f"   Confidence scores: {[f'{s:.2f}' for s in scores]}")

    result = draw_boxes(image.copy(), boxes, scores)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    # ✅ BUILD DETECTIONS ARRAY (THIS FIXES THE ERROR)
    detections = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        detections.append({
            "id": i + 1,
            "confidence": float(score),
            "box": [float(x) for x in box]
        })

    print(f"📤 Returning {len(detections)} detections\n")

    return jsonify({
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "detections": detections,              # ✅ ARRAY
        "count": len(detections),              # ✅ COUNT SEPARATE
        "model": model_name,
        "original_size": f"{image.width}x{image.height}"
    })


@app.route("/download", methods=["POST"])
def download_image():
    """Download the detection result image"""
    try:
        data = request.get_json()
        image_data = data.get("image", "")
        
        # Remove the data:image/png;base64, prefix if present
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        # Decode base64 image
        img_bytes = base64.b64decode(image_data)
        
        # Create a BytesIO object
        buf = io.BytesIO(img_bytes)
        buf.seek(0)
        
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name="detection_result.png"
        )
    except Exception as e:
        print(f"❌ Download error: {e}")
        return jsonify({"error": str(e)}), 500



# ======================
# Run App
# ======================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
