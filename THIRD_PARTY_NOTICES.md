# Third-party notices

## Deep-Live-Cam

- https://github.com/hacksider/Deep-Live-Cam
- License: GNU Affero General Public License v3.0 (`LICENSE` in this repository is the same text).
- Used for: the model files and download links (`inswapper_128_fp16.onnx`, `gfpgan-1024.onnx`) and the provider install recipes. The approach to the GFPGAN ONNX pre/post-processing and the FFHQ 5-point alignment template follows `modules/processors/frame/face_enhancer.py`. `scripts/setup_baseline.py` installs the unmodified app into `third_party/`.

## InsightFace

- https://github.com/deepinsight/insightface
- Code license: MIT.
- **Pretrained models (`buffalo_l` detection/recognition packs, `inswapper_128`): non-commercial research purposes only.** See https://github.com/deepinsight/insightface#license. This project downloads these models at setup time and does not redistribute them.

## GFPGAN

- https://github.com/TencentARC/GFPGAN
- License: Apache-2.0, with some components (StyleGAN2, DFDNet) under their own licenses. The `gfpgan-1024.onnx` export is distributed by Deep-Live-Cam.

## Python packages

The following are installed from PyPI into the project venv under their own licenses:

- OpenCV (Apache-2.0)
- ONNX Runtime (MIT)
- NumPy (BSD-3-Clause)
- PyYAML (MIT)
- pyvirtualcam (GPL-2.0)
- imageio-ffmpeg (BSD-2-Clause; the bundled ffmpeg binary is LGPL/GPL)
- tqdm (MIT/MPL-2.0)

## Sample face

`faces/source.jpg` is a StyleGAN-generated image of a person who does not exist. See `faces/README.md`.
