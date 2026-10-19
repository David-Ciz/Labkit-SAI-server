"""Synthetic fixtures exercise inference plumbing; they are not scientific results."""
import numpy as np
import pytest
import torch

from qa_ensemble.contracts import RunRequest
from qa_ensemble.frozen import FrozenPipeline


class ConstantQA(torch.nn.Module):
    def forward(self, inputs):
        return torch.full((len(inputs), 1), .8, device=inputs.device)


class ConstantPairwise:
    def predict_proba(self, inputs):
        return np.full((len(inputs), 2), .5)


class ConstantEnsemble(torch.nn.Module):
    def forward(self, inputs):
        return torch.full((len(inputs),), .5, device=inputs.device)


def pipeline():
    result = FrozenPipeline.__new__(FrozenPipeline)
    result.device = torch.device("cpu")
    result.manifest = {"dataset_identity": "new-domain", "qa": {"crop_size": 16, "raw_encoding": "preserve"},
                       "ensemble": {"max_candidates": 3, "temperature": 1., "boundary_floor": .75, "threshold": .48}}
    result.qa = ConstantQA()
    result.ranker = {"model": ConstantPairwise(), "feature_columns": ["qa_score"], "model_columns": ["diff__qa_score"], "config": {}}
    result.ensemble = ConstantEnsemble()
    result.payload = {"config": {"feature_columns": ["qa_score", "source_train_mean_iou"]},
                      "feature_columns": ["qa_score", "source_train_mean_iou"],
                      "categorical_columns": ["dataset", "candidate_source"],
                      "preprocess": {"numeric_mean": {"qa_score": .5, "source_train_mean_iou": .7},
                                     "numeric_std": {"qa_score": .1, "source_train_mean_iou": .2},
                                     "categories": {"dataset": ["old-domain"], "candidate_source": ["old-source"]}}}
    return result


def data():
    request = RunRequest.model_validate({"pipeline_id": "test", "methods": ["a", "b"],
        "region": {"original_shape_yx": [12, 20], "target_box_xyxy": [1, 1, 9, 9], "target_point_xy": [4, 4]}})
    image = np.arange(240).reshape(12, 20)
    mask = np.zeros(image.shape, dtype=bool); mask[2:7, 2:7] = True
    second = mask.copy(); second[7, 3:6] = True
    return image, {"a": mask, "b": second}, request


def test_frozen_path_scores_selects_weights_and_fuses_without_reference_labels():
    image, masks, request = data()
    result = pipeline().run(image, masks, request)
    assert result["stages"]["ensemble"]["status"] == "ok", result
    assert result["ensemble_mask"].shape == image.shape
    assert result["ensemble_mask"][4, 4]
    assert sum(row["fusion_weight"] for row in result["scores"]) == pytest.approx(1.)
    assert result["diagnostics"]["unknown_categories"]["candidate_source"] == ["a", "b"]
    assert "source_train_mean_iou" in result["diagnostics"]["imputed_ensemble_features"]


def test_later_failure_retains_qa_and_does_not_invent_ensemble_output():
    image, masks, request = data()
    instance = pipeline()
    instance.ranker["model_columns"] = ["missing_input"]
    result = instance.run(image, masks, request)
    assert len(result["scores"]) == 2
    assert result["stages"]["qa"]["status"] == "ok"
    assert result["stages"]["ranking"]["status"] == "failed"
    assert "ensemble_mask" not in result
