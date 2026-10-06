# Third-party notices

## Deep-Live-Cam

- https://github.com/hacksider/Deep-Live-Cam
- License: GNU Affero General Public License v3.0 (`LICENSE` in this repository is the same text).
- Used for: the model files and download links (`inswapper_128_fp16.onnx`, `gfpgan-1024.onnx`) and the provider install recipes. The approach to the GFPGAN ONNX pre/post-processing and the FFHQ 5-point alignment template follows `modules/processors/frame/face_enhancer.py`. `scripts/setup_baseline.py` installs the unmodified app into `third_party/`.

## InsightFace

- https://github.com/deepinsight/insightface
- Code license: MIT.
- `reveal/insight.py` is a port of part of insightface 0.7.3: the SCRFD detector, ArcFace recognition, 5-point face alignment and the inswapper wrapper. The scikit-image similarity-transform call is replaced by the same Umeyama algorithm in numpy. The port is distributed under the original MIT license:

  > Copyright (c) 2018 Jiankang Deng and Jia Guo
  >
  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  >
  > The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  >
  > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
- **Pretrained models (`buffalo_l` detection/recognition packs, `inswapper_128`): non-commercial research purposes only.** See https://github.com/deepinsight/insightface#license. This project downloads these models at setup time and does not redistribute them.

## GFPGAN

- https://github.com/TencentARC/GFPGAN
- License: Apache-2.0, with some components (StyleGAN2, DFDNet) under their own licenses. The `gfpgan-1024.onnx` export is distributed by Deep-Live-Cam.

## Python packages

The following are installed from PyPI into the project venv under their own licenses:

- OpenCV (Apache-2.0)
- PySide6 / Qt (LGPL-3.0)
- cv2_enumerate_cameras (MIT)
- certifi (MPL-2.0)
- onnx (Apache-2.0)
- ONNX Runtime (MIT)
- NumPy (BSD-3-Clause)
- PyYAML (MIT)
- pyvirtualcam (GPL-2.0)
- imageio-ffmpeg (BSD-2-Clause; the bundled ffmpeg binary is LGPL/GPL)

## Sample face

`faces/source.jpg` is a StyleGAN-generated image of a person who does not exist. See `faces/README.md`.
