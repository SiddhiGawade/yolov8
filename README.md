# YOLOv8 Webcam Demo

A small Python project demonstrating real-time webcam inference with [Ultralytics YOLOv8](https://docs.ultralytics.com/). It contains one example using a local `best.pt` checkpoint and two examples using the pretrained `yolov8l-seg.pt` segmentation model. The scripts open a desktop window to show annotated video; this repository does not contain training code.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Loads `best.pt` and draws bounding boxes, class names, and confidence scores manually. |
| `appv2.py` | Loads `yolov8l-seg.pt`, uses YOLO's `plot()` visualization, shows a frame counter, and can save annotated frames. |
| `best.py` | Minimal webcam example using `yolov8l-seg.pt` and `plot()`. |
| `best.pt` | Local model checkpoint required by `app.py`. Its classes come from the checkpoint, not the YAML file. |
| `yolov8l-seg.pt` | Pretrained YOLOv8 large segmentation weights used by `appv2.py` and `best.py`. |
| `# Classes.yaml` | Reference list of 80 COCO class names; none of the scripts reads this file. |

## Requirements

- Python (the project was set up locally with Python 3.12)
- A working webcam and a desktop environment capable of displaying OpenCV windows
- Python packages: `ultralytics`, `opencv-python`, and `numpy`

## Setup

From the project directory, create a virtual environment and install the dependencies. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install ultralytics opencv-python numpy
```

If you use another Python version, replace `py -3.12` with your installed Python command. On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Run

Run **one** of these commands from the project directory, with the virtual environment active:

```powershell
python app.py
python appv2.py
python best.py
```

- `app.py` uses `best.pt` and draws labeled detection boxes. The checkpoint must be in the project directory (or its path must be changed in the script).
- `appv2.py` and `best.py` use `yolov8l-seg.pt`. They run inference with a confidence threshold of `0.5` and display YOLO's annotated output, including segmentation where available. If the pretrained weights are missing, Ultralytics may download them on first use; an internet connection is then required.
- All three scripts use camera index `0` (the default webcam). Press `q` while the video window is focused to quit. In `appv2.py`, press `s` to save an annotated image named `detected_frame_<number>.jpg` in the current directory.

If the camera cannot be opened, check webcam permissions and whether another application is using it. To use another camera, change `cv2.VideoCapture(0)` in the script.

## GitHub backup note

The virtual environment (`.venv/`) is machine-specific; recreate it with the setup steps rather than uploading it. If you need `app.py` to work after deleting your local folder, make sure `best.pt` is included in your backup: it cannot be recovered from these scripts. `yolov8l-seg.pt` is about 92 MB, so GitHub may warn about its size; it can instead be downloaded again by Ultralytics. Check that the files you need are visible on GitHub before deleting the local copy.
