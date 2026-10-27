# DrishtiCV Models Directory

This directory stores YOLO model weights for DrishtiCV.

## Default Model
The default model is `yolo26n.pt` (Ultralytics YOLO nano model), or `yolov8n.pt` / `yolo11n.pt`.
When run for the first time with an active internet connection, Ultralytics downloads the official checkpoint automatically to the working directory or cache.

You can also place pre-downloaded `.pt` weights in this directory:
- `models/yolo26n.pt`
- `models/yolo11n.pt`
- `models/yolov8n.pt`

## Custom Models
To use custom trained YOLO models:
1. Place your trained `.pt` weights file in this folder (or anywhere on your filesystem).
2. In DrishtiCV, click **Browse Custom Model** in the left sidebar.
3. Select your `.pt` file. DrishtiCV will validate and load the classes and architecture.

For full custom model training and compatibility instructions, refer to [docs/CUSTOM_MODEL_GUIDE.md](../docs/CUSTOM_MODEL_GUIDE.md).
