# models/

Model files are downloaded here and are git-ignored. Run:

```bash
python scripts/download_models.py --buffalo            # inswapper + buffalo_l
python scripts/download_models.py --enhancer           # + GFPGAN (gfpgan-1024.onnx)
```

Expected layout:

```
models/
  inswapper_128_fp16.onnx          # or inswapper_128.onnx
  gfpgan-1024.onnx                 # optional
  insightface/models/buffalo_l/    # auto-downloaded on first run
```

The InsightFace models are licensed for non-commercial research use only.
