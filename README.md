```md
# Gesture Control Program

This project uses a webcam to detect your hand and trigger scroll macros:
- Open hand: scroll up
- Closed fist: scroll down

It uses MediaPipe Hands for landmark detection and a pretrained TFLite keypoint classifier (from kinivi/hand-gesture-recognition-mediapipe) for gesture classification.

## Features

- Real-time hand tracking with MediaPipe
- Gesture classification via TFLite model
- Debounce and rate limiting to prevent input spam
- Windows-friendly build output using PyInstaller

## Project Structure

```

Gesture-Control-Program/
main.py
gesture_classifier.py
keypoint_preprocess.py
model/
keypoint_classifier/
keypoint_classifier.tflite
keypoint_classifier_label.csv
requirements.txt

````

## Requirements

- Windows 10/11
- Python 3.10 recommended
- Webcam

## Setup

From the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
````

Run:

```powershell
python main.py
```

Press `q` to quit.

## Safety

* PyAutoGUI failsafe is enabled. Moving your mouse cursor to a screen corner will stop the program if anything goes wrong.

## Build an EXE (PyInstaller)

Install PyInstaller:

```powershell
python -m pip install pyinstaller pyinstaller-hooks-contrib
```

Build (recommended one-folder output):

```powershell
pyinstaller --noconfirm --clean --name "GestureController" --onedir --windowed --collect-all mediapipe --add-data "model\keypoint_classifier\keypoint_classifier.tflite;model\keypoint_classifier" --add-data "model\keypoint_classifier\keypoint_classifier_label.csv;model\keypoint_classifier" main.py
```

Output:

* `dist\GestureController\GestureController.exe`

## Configuration

Adjust these values in `main.py` to tune behavior:

* `STABLE_FRAMES`: higher reduces flicker (typical 6–12)
* `SCROLL_INTERVAL`: higher slows repeated scrolling
* `SCROLL_STEP`: lower reduces scroll strength per tick

## Troubleshooting

### MediaPipe error: missing mediapipe/modules in EXE

Rebuild using `--collect-all mediapipe` (already included in the build command above).

### Camera does not open

* Close apps using the camera (Zoom/Teams/OBS/browser)
* Try a different camera index (0/1/2)
* Use the DirectShow backend on Windows (`cv2.CAP_DSHOW`)

## Credits

* Gesture classifier model and dataset format based on:
  kinivi/hand-gesture-recognition-mediapipe

```
::contentReference[oaicite:0]{index=0}
```
