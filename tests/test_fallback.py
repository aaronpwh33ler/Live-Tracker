"""SafeSession steps down to the next provider setup when acceleration fails."""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reveal import insight  # noqa: E402


def _tiny_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_coreml_chain_steps_down_to_cpu():
    chain = insight.fallback_chain([("CoreMLExecutionProvider", {}), "CPUExecutionProvider"])
    assert [insight._describe(c) for c in chain] == ["CoreML", "CoreML (GPU only)", "CPU"]
    assert insight.fallback_chain(["CPUExecutionProvider"]) == [["CPUExecutionProvider"]]
    assert insight.fallback_chain(["CUDAExecutionProvider", "CPUExecutionProvider"])[-1] == ["CPUExecutionProvider"]


def test_runtime_failure_falls_back(tmp_path, monkeypatch, capsys):
    model = tmp_path / "m.onnx"
    _tiny_model(model)
    # Pretend the first two setups are accelerated ones that fail while running.
    monkeypatch.setattr(insight, "fallback_chain",
                        lambda providers: [["FakeAccel"], ["FakeAccel2"], ["CPUExecutionProvider"]])
    real = insight.onnxruntime.InferenceSession

    class Broken:
        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            raise RuntimeError("Error executing model")

    def make(path, sess_options=None, providers=None):
        return Broken() if providers[0] != "CPUExecutionProvider" else real(path, sess_options=sess_options,
                                                                              providers=providers)

    monkeypatch.setattr(insight.onnxruntime, "InferenceSession", make)
    monkeypatch.setattr(insight, "_provider_name", lambda p: p[0] if isinstance(p, tuple) else p)
    s = insight.SafeSession(model, ["whatever"])
    out = s.run(None, {"x": np.array([[-1.0, 2.0]], np.float32)})[0]
    assert out.tolist() == [[0.0, 2.0]]
    assert s.level == 2
    log = capsys.readouterr().out
    assert log.count("failed while running") == 2 and "m.onnx" in log


def test_cpu_errors_are_not_swallowed(tmp_path):
    model = tmp_path / "m.onnx"
    _tiny_model(model)
    s = insight.SafeSession(model, ["CPUExecutionProvider"])
    try:
        s.run(None, {"x": np.zeros((3, 3), np.float32)})  # wrong shape
    except Exception:
        return
    raise AssertionError("expected an error")
