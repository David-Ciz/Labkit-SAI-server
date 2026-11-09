"""Regressions for independent candidate sets and corrected-model inference."""
import numpy as np
import pandas as pd
import pytest
import torch

from qa_ensemble.frozen import ensemble_matrix, ensemble_raw_scores, relative_features


def rows():
    return pd.DataFrame([
        dict(dataset="A", fold=fold, campaign_number="02", split="test",
             image_id="02_t001", label=label, center_x=4., center_y=4.,
             candidate_source=f"source-{i}", qa_score=value, crop_width=16)
        for fold in ["fold-1", "fold-2"] for label in [1, 2]
        for i, value in enumerate([.1, .5, .9])
    ])


def payload():
    return {"candidate_grouping_version": 2,
            "config": {"feature_columns": ["qa_score", "crop_width"]}}


def test_corrected_context_separates_labels_and_folds_and_discards_stale_features():
    source = rows()
    source["candidate_set_rank__qa_score"] = -999
    combined = relative_features(source, payload())
    separate = pd.concat([relative_features(group, payload())
                          for _, group in source.groupby(["fold", "label"])]).sort_index()
    assert combined.ranker_cluster_id.nunique() == 4
    assert set(combined.ranker_cluster_size) == {3}
    assert combined.columns.is_unique
    np.testing.assert_allclose(combined.candidate_set_rank__crop_width, 2/3)
    pd.testing.assert_frame_equal(combined, separate)
    pd.testing.assert_frame_equal(combined, relative_features(combined, payload()))
    changed = source.copy()
    changed.loc[changed.label.eq(2), "qa_score"] = 100
    rescored = relative_features(changed, payload())
    pd.testing.assert_series_equal(combined.loc[combined.label.eq(1), "candidate_set_centered__qa_score"],
                                   rescored.loc[rescored.label.eq(1), "candidate_set_centered__qa_score"])


def test_inference_preprocessing_is_float32_and_model_context_is_per_set():
    data = pd.DataFrame({"qa_score": [1., 3., 100., 200.],
                         "dataset": ["A"] * 4,
                         "ranker_cluster_id": ["first", "first", "second", "second"]})
    model_payload = {"feature_columns": ["qa_score"], "categorical_columns": ["dataset"],
                     "preprocess": {"numeric_mean": {"qa_score": 0.},
                                    "numeric_std": {"qa_score": 1.}, "categories": {"dataset": ["A"]}}}
    assert ensemble_matrix(data, model_payload).dtype == np.float32

    class SetMean(torch.nn.Module):
        def forward(self, inputs):
            assert inputs.dtype == torch.float32
            return inputs[:, 0].mean().expand(len(inputs))

    scores = ensemble_raw_scores(data, model_payload, SetMean())
    np.testing.assert_allclose(scores, [2., 2., 150., 150.])


def test_unknown_artifact_semantics_are_rejected():
    bad = payload()
    bad["candidate_grouping_version"] = 3
    with pytest.raises(ValueError, match="Unsupported candidate-grouping"):
        relative_features(rows(), bad)
