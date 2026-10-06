# Troubleshooting

## onnxruntime provider conflicts

**Symptoms:**

- `[provider] CUDAExecutionProvider is not available in this onnxruntime build`
- `requested CUDAExecutionProvider but onnxruntime is running on CPUExecutionProvider`
- unexpectedly slow runs

`onnxruntime`, `onnxruntime-gpu`, `onnxruntime-directml`, `onnxruntime-openvino` and the old `onnxruntime-silicon` all install the same `onnxruntime` Python module. If two are installed, whichever was installed last wins, often silently. Fix:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-openvino onnxruntime-silicon onnxruntime-coreml
pip install -r requirements-<provider>.txt
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

The printed list must contain the provider `scripts/check_env` picked.

**CUDA is listed but it still runs on CPU.** The CUDA/cuDNN libraries failed to load:

- Check that `nvidia-smi` works, which means the driver is installed.
- `requirements-cuda.txt` installs CUDA 12 and cuDNN 9 into the venv via `onnxruntime-gpu[cuda,cudnn]`, and `reveal_cam` calls `onnxruntime.preload_dlls()` to load them.
- If you use a system CUDA instead, it must be CUDA 12.x with cuDNN 9.x, and its `bin`/`lib` directories must be on `PATH` / `LD_LIBRARY_PATH`.
- Run once with `--provider cuda` and read the onnxruntime warning printed at startup. It names the missing library.

**OpenVINO:** `onnxruntime-openvino` newer than 1.21.0 must be installed together with the matching `openvino` version (see the pairing table in the Deep-Live-Cam README).

**Mixed OpenCV packages.** `opencv-python-headless` has no preview window. If both it and `opencv-python` are installed, you may get "no GUI available". Fix with `pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python==4.14.0.94`.

## The app window or launcher

- **Mac: "Reveal Cam.command" Not Opened / "Apple could not verify… is free of malware".** macOS blocks double-clicked scripts downloaded from the internet. Do one of these:
  - Click **Done**, open **System Settings → Privacy & Security**, scroll down and click **Open Anyway**. On macOS 15 (Sequoia) and later, right-click → Open no longer bypasses this.
  - In Terminal, `cd` into the folder and run `xattr -dr com.apple.quarantine .` once. Double-clicking works from then on.
  - Or skip the launcher: in Terminal, inside the folder, run `./start.sh`.
- **Mac: "Permission denied" when double-clicking** (some unzip tools drop the executable flag). In Terminal, run `chmod +x "Reveal Cam.command" start.sh` inside the folder.
- **"Reveal Cam needs Python 3.11, 3.12, 3.13 or 3.14"**, or setup fails with **"No matching distribution found for PySide6"** or onnxruntime: your Python is too old, or too new (a pre-release such as 3.15). Install Python 3.13 from https://www.python.org/downloads/; it can sit alongside other versions. On Windows, tick "Add python.exe to PATH". Then start Reveal Cam again. It picks 3.13 automatically and rebuilds its `.venv`.
- **Setup fails halfway** (network drop, full disk): double-click again; setup picks up where it left off. To start completely fresh, delete the `.venv` folder.
- **Download fails with `CERTIFICATE_VERIFY_FAILED` on a Mac**: run "Install Certificates.command" from your Python folder in Applications, then try again.
- **Linux: the window doesn't open, with "Could not load the Qt platform plugin xcb".** Install Qt's X11 libraries, for example on Debian/Ubuntu: `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-xkb1`.
- **The camera list only shows "Default camera / Camera 2 / Camera 3"**: the app couldn't read camera names. Try each one; the right one opens your webcam.
- **The app closes the camera window right away**: click "Start camera" again and read the error box. Its "Show Details" button has the full log.

## Apple Silicon: CoreML errors

**Symptom:** a message like `[provider] det_10g.onnx: CoreML failed while running (… Error executing model …). Switching to CoreML (GPU only).`

Apple's CoreML acceleration can refuse some models, sometimes only once they start running. Reveal Cam then retries that model with CoreML on the GPU only, and finally on the CPU, so it keeps working, just slower for that model. Which models fell back is printed in the Terminal window. Please report it: it tells us which model needs a CoreML-specific fix. To skip CoreML entirely, set `execution_provider: cpu` in `config.yaml`.

## Camera permission prompts

- **macOS** asks for camera permission per *terminal app* (Terminal, iTerm, VS Code), not per Python script. The first time, approve the prompt. If you dismissed it, open **System Settings → Privacy & Security → Camera**, enable your terminal app, then **quit and reopen** the terminal. Running from an IDE's integrated terminal uses the IDE's permission.
- **Windows:** **Settings → Privacy & security → Camera**: turn on "Let desktop apps access your camera".
- **Linux:** your user needs access to `/dev/video*` (usually the `video` group: `sudo usermod -aG video $USER`, then log out and in again).
- **"Could not open camera 0"**: another app (Zoom, Teams, OBS, a browser tab) may be holding the camera, or your webcam has a different index. Try `--camera 1` or `--camera 2`.

## "No face detected"

- **Lighting:** light your face from the front. Strong backlight (a window behind you) and dark rooms both hurt detection.
- **Angle:** look roughly at the camera. Strong profile views (more than about 60°) and faces tilted far down or up are often missed.
- **Distance:** a face filling almost the whole frame is *harder* to detect, not easier. Sit back so your head takes up roughly ⅓ of the frame height. Very small faces in high-resolution video may need a larger `--det-width` (for example 800 or 960).
- **Occlusion:** big sunglasses, masks, or a hand over the face cause misses.
- **Detection threshold:** lower `detection.min_score` (for example `--set detection.min_score=0.35`) to accept weaker detections.
- **Source image:** `reveal_cam` pads and retries tight portrait crops automatically. The stock Deep-Live-Cam app does not. If it says "No face detected in source image", give it a less tightly cropped photo.

## Low frame rate

Check the FPS overlay (`i`). It shows per-stage timings (det / swap / enh) so you can see which part is slow.

1. **Turn off the enhancer** (`e`, or `--no-enhancer`). GFPGAN costs more than the swap itself.
2. **Lower the detection scale:** `--det-width 480` or `320`.
3. **Increase skip-N:** `--skip-n 2` or `3` detects every Nth frame and reuses the boxes in between.
4. **Lower the camera resolution:** `--width 960 --height 540`.
5. **Check the provider** in the overlay. If it says `CPU` on a GPU machine, see the provider-conflict section above.
6. **Only swap one face:** `swap.all_faces: false` (the default).
7. **Close the window when you don't need it.** No swap runs while the reveal is closed.
8. On CPU-only machines, use **offline mode**: `--input clip.mp4 --output out.mp4`.

`python scripts/benchmark.py --video clip.mp4` prints the per-stage costs on your machine.

## Recording / audio

- **The recording is not H.264** or the console shows `ffmpeg not found`: install ffmpeg system-wide, or reinstall `imageio-ffmpeg` in the venv. Without either, OpenCV writes an `mp4v` file, which some players can't open.
- **Offline output has no audio:** the input has no audio track, or the remux failed. The console says which.

## Virtual camera

`[virtualcam] skipped: ...` means no virtual camera device was found:

- **Windows/macOS:** install OBS Studio 26+, then start and stop its Virtual Camera once to register it.
- **Linux:** `sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback devices=1 exclusive_caps=1`.
