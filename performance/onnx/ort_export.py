"""
Step 97: ONNX Runtime optimisation for production inference.

ONNX (Open Neural Network Exchange) provides:
  - Hardware-agnostic inference: same model runs on CPU, GPU, ARM
  - ~2–5× speedup over PyTorch eager mode on CPU
  - Graph optimisations: operator fusion, constant folding, layout transforms
  - INT8 quantisation: 4× memory reduction, 2–3× CPU speedup

Export pipeline per model:
  PyTorch (fp32) → ONNX → ORT optimised → ORT INT8 quantised

Latency targets:
  Text (512 tokens, CPU):  < 200ms  (student < 50ms)
  Image (224×224, CPU):    < 100ms  (student < 20ms)
  Audio (30s chunk, CPU):  < 500ms  (student < 100ms)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Latency targets in milliseconds (Step 102)
LATENCY_TARGETS_MS = {
    "text_deep":    2000,    # sub-2s for text (full ensemble)
    "text_fast":    200,     # pre-screen pass
    "image_deep":   5000,
    "image_fast":   100,
    "audio_deep":   10000,   # sub-10s for media
    "audio_fast":   500,
    "video_deep":   10000,
    "video_fast":   1000,
    "code_deep":    2000,
    "code_fast":    100,
}


# ── ONNX export ────────────────────────────────────────────────

def export_to_onnx(
    model:        Any,
    dummy_input:  Any,
    output_path:  Path,
    input_names:  list[str] = ["input"],
    output_names: list[str] = ["output"],
    dynamic_axes: dict | None = None,
    opset:        int = 17,
) -> Path:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model:        PyTorch nn.Module (in eval mode)
        dummy_input:  Representative input tensor or dict of tensors
        output_path:  Where to write the .onnx file
        dynamic_axes: Dict of {name: {dim: "name"}} for variable-length dims
        opset:        ONNX opset version (17 = latest stable)

    Returns:
        Path to the exported .onnx file
    """
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Default dynamic axes for variable-length sequence support
    if dynamic_axes is None and isinstance(dummy_input, dict):
        dynamic_axes = {name: {0: "batch", 1: "sequence"}
                        for name in input_names}
    elif dynamic_axes is None:
        dynamic_axes = {input_names[0]:  {0: "batch", 1: "sequence"},
                        output_names[0]: {0: "batch"}}

    log.info("onnx_export_start",
             output=str(output_path), opset=opset)

    with torch.no_grad():
        if isinstance(dummy_input, dict):
            torch.onnx.export(
                model,
                kwargs=dummy_input,
                f=str(output_path),
                opset_version=opset,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes={k: {0: "batch", 1: "seq"} for k in dummy_input},
                do_constant_folding=True,
            )
        else:
            torch.onnx.export(
                model,
                args=(dummy_input,),
                f=str(output_path),
                opset_version=opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )

    # Verify the export
    try:
        import onnx  # type: ignore
        m = onnx.load(str(output_path))
        onnx.checker.check_model(m)
        log.info("onnx_validation_passed", path=str(output_path))
    except ImportError:
        log.warning("onnx_not_installed — skipping validation")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("onnx_exported", path=str(output_path), size_mb=round(size_mb, 2))
    return output_path


# ── ORT optimisation ───────────────────────────────────────────

def optimise_onnx(
    onnx_path:   Path,
    output_path: Path | None = None,
    level:       str = "all",   # "basic" | "extended" | "all"
) -> Path:
    """
    Apply ONNX Runtime graph optimisations:
      basic    — constant folding, redundant node elimination
      extended — operator fusion (Conv+BN, GELU, etc.)
      all      — full ORT optimisations including layout transforms

    Returns path to the optimised model.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pip install onnxruntime") from exc

    out = output_path or onnx_path.with_suffix(".opt.onnx")

    opt_level = {
        "basic":    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all":      ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }.get(level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = opt_level
    sess_options.optimized_model_filepath = str(out)

    # Trigger optimisation by creating a session (ORT optimises on load)
    ort.InferenceSession(str(onnx_path), sess_options=sess_options)
    log.info("onnx_optimised", output=str(out), level=level)
    return out


def quantise_onnx_int8(
    onnx_path:   Path,
    output_path: Path | None = None,
    calibration_data: list[np.ndarray] | None = None,
) -> Path:
    """
    Apply INT8 post-training quantisation to an ONNX model.

    Dynamic quantisation (no calibration data needed): quantises
    weights to INT8, activations computed in FP32. Good default.

    Static quantisation (with calibration data): both weights and
    activations INT8. Better for throughput-critical deployments.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pip install onnxruntime") from exc

    out = output_path or onnx_path.with_suffix(".int8.onnx")

    if calibration_data is None:
        # Dynamic quantisation — no calibration needed
        quantize_dynamic(
            str(onnx_path),
            str(out),
            weight_type=QuantType.QInt8,
        )
        log.info("onnx_quantised_dynamic", output=str(out))
    else:
        # Static quantisation with calibration
        from onnxruntime.quantization import (  # type: ignore
            quantize_static, CalibrationDataReader, QuantType,
        )

        class _CalibReader(CalibrationDataReader):
            def __init__(self, data: list[np.ndarray]) -> None:
                self._data = iter(data)
            def get_next(self):
                try:
                    return {"input": next(self._data)[np.newaxis]}
                except StopIteration:
                    return None

        quantize_static(
            str(onnx_path), str(out),
            calibration_data_reader=_CalibReader(calibration_data),
            weight_type=QuantType.QInt8,
        )
        log.info("onnx_quantised_static", output=str(out))

    size_orig = onnx_path.stat().st_size / (1024 * 1024)
    size_q    = out.stat().st_size / (1024 * 1024)
    log.info("quantisation_compression",
             original_mb=round(size_orig, 2),
             quantised_mb=round(size_q, 2),
             ratio=round(size_orig / max(size_q, 0.1), 2))
    return out


# ── ORT inference session ──────────────────────────────────────

class ONNXDetector:
    """
    Production inference wrapper using ONNX Runtime.
    Loads an optimised/quantised model and runs inference.
    """

    def __init__(
        self,
        model_path:    Path,
        providers:     list[str] | None = None,
        num_threads:   int = 4,
    ) -> None:
        self._model_path = model_path
        self._providers  = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._num_threads = num_threads
        self._session: Any = None

    def load(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pip install onnxruntime") from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._num_threads
        opts.inter_op_num_threads = 1   # single graph for latency
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=opts,
            providers=self._providers,
        )
        log.info("ort_session_loaded",
                 path=str(self._model_path),
                 providers=self._session.get_providers())

    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run inference. Returns raw logits as numpy array."""
        if self._session is None:
            raise RuntimeError("Call load() first")
        outputs = self._session.run(None, inputs)
        return outputs[0]

    def predict_proba(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run inference and apply softmax. Returns probabilities."""
        logits = self.predict(inputs)
        exp    = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def benchmark(
        self,
        dummy_inputs: dict[str, np.ndarray],
        n_runs: int = 100,
        warmup: int = 10,
    ) -> dict[str, float]:
        """
        Benchmark latency over n_runs iterations.
        Returns mean, p50, p95, p99 latency in milliseconds.
        """
        # Warmup
        for _ in range(warmup):
            self.predict(dummy_inputs)

        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(dummy_inputs)
            times.append((time.perf_counter() - t0) * 1000)

        times_arr = np.array(times)
        return {
            "mean_ms":   round(float(times_arr.mean()), 2),
            "p50_ms":    round(float(np.percentile(times_arr, 50)), 2),
            "p95_ms":    round(float(np.percentile(times_arr, 95)), 2),
            "p99_ms":    round(float(np.percentile(times_arr, 99)), 2),
            "min_ms":    round(float(times_arr.min()), 2),
            "max_ms":    round(float(times_arr.max()), 2),
            "n_runs":    n_runs,
        }


# ── Full export pipeline ──────────────────────────────────────

def export_all_detectors(
    checkpoints_base: Path,
    output_base:      Path,
    quantise:         bool = True,
) -> dict[str, Path]:
    """
    Export all five detectors to optimised ONNX format.
    Returns {detector_name: onnx_path}.
    """
    results: dict[str, Path] = {}

    detectors = [
        ("text",  checkpoints_base / "text_detector/checkpoints/transformer/phase3"),
        ("image", checkpoints_base / "image_detector/checkpoints/phase3"),
        ("audio", checkpoints_base / "audio_detector/checkpoints/phase3"),
        ("video", checkpoints_base / "video_detector/checkpoints/phase3"),
        ("code",  checkpoints_base / "code_detector/checkpoints/transformer/phase3"),
    ]

    for name, ckpt_dir in detectors:
        out_dir = output_base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = out_dir / f"{name}_detector.onnx"
        opt_path  = out_dir / f"{name}_detector.opt.onnx"
        int8_path = out_dir / f"{name}_detector.int8.onnx"

        if not ckpt_dir.exists():
            log.warning("checkpoint_not_found", detector=name, path=str(ckpt_dir))
            continue

        try:
            # Load and export
            model, dummy = _load_model_for_export(name, ckpt_dir)
            if model is None:
                continue

            export_to_onnx(model, dummy, onnx_path)
            optimise_onnx(onnx_path, opt_path)

            if quantise:
                quantise_onnx_int8(opt_path, int8_path)
                results[name] = int8_path
            else:
                results[name] = opt_path

            log.info("detector_exported", name=name)
        except Exception as exc:
            log.error("detector_export_failed", name=name, error=str(exc))

    return results


def _load_model_for_export(
    name: str, ckpt_dir: Path,
) -> tuple[Any, Any]:
    """Load a detector model and create a dummy input for ONNX export."""
    import torch
    try:
        if name in ("text", "code"):
            from transformers import AutoModelForSequenceClassification  # type: ignore
            model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
            model.eval()
            dummy = {
                "input_ids":      torch.zeros(1, 512, dtype=torch.long),
                "attention_mask": torch.ones(1, 512, dtype=torch.long),
            }
            return model, dummy

        elif name == "image":
            try:
                import timm  # type: ignore
                model = timm.create_model("efficientnet_b4", num_classes=2)
                model.load_state_dict(torch.load(ckpt_dir / "efficientnet.pt",
                                                   map_location="cpu"))
                model.eval()
                dummy = torch.randn(1, 3, 224, 224)
                return model, dummy
            except Exception:
                return None, None

        elif name == "audio":
            model = build_audio_student()
            if model:
                model.eval()
                dummy = torch.randn(1, 39, 300)  # [batch, mfcc, time]
            return model, dummy

    except Exception as exc:
        log.warning("model_load_failed", name=name, error=str(exc))

    return None, None
