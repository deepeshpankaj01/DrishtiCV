# DrishtiCV Custom Model Integration Guide

**Application:** DrishtiCV — Real-Time Object Detection & Tracking  
**Developer:** Deepesh Kumar Pankaj  
**Tagline:** *See Objects. Track Movement. Understand the Scene.*

---

## 1. Overview & Scope Limitation

By default, DrishtiCV loads the official pretrained **Ultralytics YOLO26n** checkpoint (`yolo26n.pt`), which detects the standard **80 COCO object classes** (people, vehicles, animals, common household items).

> [!IMPORTANT]
> The default model is trained on everyday objects. It **cannot** detect semiconductor wafer defects, printed circuit board (PCB) soldering flaws, or specialised electronic micro-components out of the box. To perform industrial or electronics inspection, you must train a domain-specific YOLO model on a labeled dataset and load your custom weights using the steps documented below.

---

## 2. Model Compatibility Requirements

DrishtiCV supports any object detection model exported as a PyTorch checkpoint (`.pt`) using the Ultralytics architecture family:
- **YOLO26** (`yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`)
- **YOLO11** (`yolo11n.pt`, `yolo11s.pt`, etc.)
- **YOLOv8** (`yolov8n.pt`, `yolov8s.pt`, etc.)

### Required Output Format
The model must be trained for the **`detect`** task. Models exported for classification (`-cls`), pose (`-pose`), or segmentation (`-seg`) will run detection on their bounding boxes if supported, but standard 2D bounding-box detection models are recommended for full tracker and analytics compatibility.

---

## 3. How to Load a Custom Model in DrishtiCV

### Method A: Through the Desktop GUI
1. Launch DrishtiCV using `run.bat` or `python app.py`.
2. Locate the **Object Detection Model** panel in the left sidebar.
3. Click the **Browse Custom .pt File...** button.
4. Select your custom trained `.pt` file from your disk.
5. DrishtiCV will automatically:
   - Validate the model architecture.
   - Extract the custom class names directly from the checkpoint's internal metadata (`model.names`).
   - Populate the class filter dialog with your custom class labels.
   - Update the status bar and model badge to confirm successful loading.

### Method B: Placing Weights in the `models/` Directory
1. Copy your trained weights file into the project's `models/` directory:
   ```
   drishticv/
   └── models/
       ├── yolo26n.pt
       └── pcb_component_detector.pt
   ```
2. The model will appear in the dropdown or can be loaded by filename.

---

## 4. How to Train a Domain-Specific Model

If you are developing a custom model for a specialized domain (e.g., electronic component inspection, lab automation, or industrial monitoring), follow this workflow:

### Step 1: Collect & Annotate Images
1. Gather at least 200–1,000 images representing your target objects under varied lighting and angles.
2. Label bounding boxes using tools such as CVAT, Label Studio, or Roboflow in **YOLO format** (normalized `class x_center y_center width height`).

### Step 2: Prepare the Dataset Configuration (`dataset.yaml`)
Create a YAML file describing dataset splits and classes:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: resistor_smd
  1: capacitor_ceramic
  2: ic_soic8
  3: solder_bridge_defect
```

### Step 3: Train Using Ultralytics
In your Python training environment:

```python
from ultralytics import YOLO

# Initialize from nano pretrained backbone for fast transfer learning
model = YOLO("yolo26n.pt")

# Train on custom dataset
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="custom_electronics_model"
)
```

The resulting weights will be saved to:
`runs/detect/custom_electronics_model/weights/best.pt`

### Step 4: Verify and Load into DrishtiCV
Copy `best.pt` to `models/` and select it in DrishtiCV. DrishtiCV will display your specific class names (`resistor_smd`, `solder_bridge_defect`) and track them in real time across the video workspace.
