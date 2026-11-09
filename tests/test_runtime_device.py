import pytest
import torch

from qa_ensemble.runtime_device import model_device, select_device


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device().type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device().type == "cpu"


def test_explicit_mps_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Metal/MPS was requested"):
        select_device("mps")


def test_per_model_cpu_override_is_respected(monkeypatch):
    monkeypatch.setenv("LABKIT_DEVICE", "mps")
    monkeypatch.setenv("LABKIT_DEVICE_MASKRCNNV1COCO", "cpu")
    assert model_device("maskrcnnv1COCO").type == "cpu"
