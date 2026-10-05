"""Pick an ONNX Runtime execution provider."""

from __future__ import annotations

import json
import platform
import subprocess
import sys

from .config import PROJECT_ROOT

ALIASES = {
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

REPORT_PATH = PROJECT_ROOT / ".env_report.json"
CHECK_ENV = PROJECT_ROOT / "scripts" / "check_env"


def _from_report() -> str | None:
    if not REPORT_PATH.exists():
        try:
            subprocess.run([sys.executable, str(CHECK_ENV), "--json"], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        return json.loads(REPORT_PATH.read_text())["provider"]
    except (OSError, ValueError, KeyError):
        return None


def resolve_providers(requested: str) -> list:
    """Return the providers list to hand to onnxruntime/insightface.

    `requested` is 'auto', an alias, or a full provider name. Falls back to CPU
    (with a warning) if the provider isn't available in the installed
    onnxruntime build.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    req = (requested or "auto").strip()
    if req.lower() == "auto":
        name = _from_report() or "CPUExecutionProvider"
    else:
        name = ALIASES.get(req.lower(), req)

    if name == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()  # load CUDA/cuDNN from the pip-installed nvidia-* packages
        except Exception as e:
            print(f"[provider] could not preload CUDA libraries: {e}")

    if name not in available:
        print(f"[provider] {name} is not available in this onnxruntime build "
              f"(available: {available}). Falling back to CPUExecutionProvider. "
              f"See TROUBLESHOOTING.md -> 'onnxruntime provider conflicts'.")
        name = "CPUExecutionProvider"

    if name == "CoreMLExecutionProvider" and platform.machine() == "arm64":
        providers: list = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                           "CPUExecutionProvider"]
    elif name == "CPUExecutionProvider":
        providers = ["CPUExecutionProvider"]
    else:
        providers = [name, "CPUExecutionProvider"]
    return providers


def provider_name(providers: list) -> str:
    p = providers[0]
    return p[0] if isinstance(p, tuple) else p
