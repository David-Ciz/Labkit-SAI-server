"""Explicit CUDA/MPS/CPU selection shared by native and container runtimes."""
import logging
import os
import platform

import torch

logger = logging.getLogger(__name__)


def select_device(requested="auto"):
    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError("LABKIT_DEVICE must be auto, cuda, mps or cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose an available device explicitly")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal/MPS was requested but is unavailable; choose cpu explicitly for CPU testing")
    return device


def model_device(model):
    override = "LABKIT_DEVICE_" + model.upper()
    device = select_device(os.getenv(override, os.getenv("LABKIT_DEVICE", "auto")))
    logger.info("%s inference device: %s (configuration: %s)", model, device,
                override if override in os.environ else "LABKIT_DEVICE")
    return device


def runtime_report():
    return {"platform": platform.system(), "architecture": platform.machine(), "torch": torch.__version__,
            "requested_device": os.getenv("LABKIT_DEVICE", "auto"),
            "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available(),
            "models": {model: str(model_device(model)) for model in
                       ("cellpose", "sam2", "instanseg", "maskrcnnv1COCO")},
            "qa_device": str(select_device(os.getenv("LABKIT_QA_DEVICE", "cpu"))),
            "mps_operator_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"}
