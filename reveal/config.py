"""Settings: defaults <- config.yaml <- command-line overrides."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30, "mirror": True},
    "source_face": "faces/source.jpg",
    "execution_provider": "auto",
    "models_dir": "models",
    "swap": {"enabled": True, "all_faces": False},
    "enhancer": {"enabled": False},
    "detection": {"width": 640, "skip_n": 1, "min_score": 0.5},
    "mask": {
        "mode": "rect",
        "feather": 25,
        "border": {"enabled": True, "thickness": 2, "color": [255, 255, 255]},
        "rect": [0.30, 0.15, 0.70, 0.85],
        "quad": [[0.33, 0.10], [0.72, 0.18], [0.66, 0.92], [0.27, 0.84]],
        "face": {"padding": 1.6, "smoothing": 0.4, "hold_frames": 15},
    },
    "animation": {
        "style": "wipe",
        "direction": "right",
        "frames": 20,
        "start_open": True,
        "keyframes": [],
    },
    "output": {
        "preview": True,
        "fullscreen": False,
        "show_fps": True,
        "record_dir": "recordings",
        "crf": 20,
        "virtual_cam": False,
    },
}

MASK_MODES = ("rect", "quad", "face", "full")
ANIM_STYLES = ("wipe", "slide", "scale", "fade")
DIRECTIONS = ("left", "right", "up", "down")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def resolve_path(p: str | None) -> Path | None:
    """Relative paths are relative to the project root, not the cwd."""
    if not p:
        return None
    path = Path(p).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="reveal_cam.py",
        description="Real-time face swap that shows through a 'reveal window'.",
    )
    ap.add_argument("--config", default="config.yaml", help="YAML settings file (default: config.yaml)")
    ap.add_argument("--source", "-s", help="source face image (fictional or consenting person)")
    ap.add_argument("--input", "-i", help="offline mode: input video file instead of the webcam")
    ap.add_argument("--output", "-o", help="offline mode: output video file")
    ap.add_argument("--camera", help="camera index (or a video file / stream URL, for testing)")
    ap.add_argument("--width", type=int, help="camera capture width")
    ap.add_argument("--height", type=int, help="camera capture height")
    ap.add_argument("--provider", help="execution provider: auto, cuda, coreml, dml, openvino, cpu, "
                                       "or a full name like CUDAExecutionProvider")
    ap.add_argument("--enhancer", dest="enhancer", action="store_true", default=None, help="enable GFPGAN")
    ap.add_argument("--no-enhancer", dest="enhancer", action="store_false", help="disable GFPGAN")
    ap.add_argument("--mask", choices=MASK_MODES, help="mask mode")
    ap.add_argument("--feather", type=int, help="mask edge feather in pixels")
    ap.add_argument("--det-width", type=int, help="detection width in pixels (e.g. 320, 480, 640)")
    ap.add_argument("--skip-n", type=int, help="run detection every N frames")
    ap.add_argument("--mirror", dest="mirror", action="store_true", default=None, help="mirror the frame")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", help="don't mirror the frame")
    ap.add_argument("--virtual-cam", dest="virtual_cam", action="store_true", default=None,
                    help="send output to a virtual camera")
    ap.add_argument("--preview", dest="preview", action="store_true", default=None,
                    help="show a preview window (always on in live mode; opt-in for offline mode)")
    ap.add_argument("--no-preview", dest="preview", action="store_false", help="don't open a preview window")
    ap.add_argument("--record", action="store_true", help="live mode: start recording immediately")
    ap.add_argument("--stdin-control", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max-frames", type=int, help="stop after N frames (testing/benchmarks)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value, e.g. --set mask.border.enabled=false "
                         "(value is parsed as YAML)")
    return ap


def load_config(argv: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    args = build_arg_parser().parse_args(argv)

    cfg = copy.deepcopy(DEFAULTS)
    cfg_path = resolve_path(args.config)
    if cfg_path and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    elif args.config != "config.yaml":
        raise SystemExit(f"Config file not found: {args.config}")

    simple = {
        "source": "source_face",
        "camera": "camera.index",
        "width": "camera.width",
        "height": "camera.height",
        "provider": "execution_provider",
        "enhancer": "enhancer.enabled",
        "mask": "mask.mode",
        "feather": "mask.feather",
        "det_width": "detection.width",
        "skip_n": "detection.skip_n",
        "mirror": "camera.mirror",
        "virtual_cam": "output.virtual_cam",
        "preview": "output.preview",
    }
    for attr, dotted in simple.items():
        val = getattr(args, attr)
        if val is not None:
            set_dotted(cfg, dotted, val)

    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, raw = item.split("=", 1)
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))

    validate(cfg)
    return cfg, args


def validate(cfg: dict) -> None:
    if cfg["mask"]["mode"] not in MASK_MODES:
        raise SystemExit(f"mask.mode must be one of {MASK_MODES}")
    if cfg["animation"]["style"] not in ANIM_STYLES:
        raise SystemExit(f"animation.style must be one of {ANIM_STYLES}")
    if cfg["animation"]["direction"] not in DIRECTIONS:
        raise SystemExit(f"animation.direction must be one of {DIRECTIONS}")
    if len(cfg["mask"]["rect"]) != 4 or len(cfg["mask"]["quad"]) != 4:
        raise SystemExit("mask.rect needs 4 numbers; mask.quad needs 4 [x, y] points")
    cfg["detection"]["skip_n"] = max(1, int(cfg["detection"]["skip_n"]))
    cfg["detection"]["width"] = max(128, int(cfg["detection"]["width"]))
    cfg["mask"]["feather"] = max(0, int(cfg["mask"]["feather"]))
    cfg["animation"]["frames"] = max(1, int(cfg["animation"]["frames"]))
    for kf in cfg["animation"]["keyframes"]:
        if "t" not in kf or kf.get("action") not in ("open", "close"):
            raise SystemExit("each animation keyframe needs {t: seconds, action: open|close}")
