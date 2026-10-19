#!/usr/bin/env python3
"""Compare the new inference definitions with original code and saved outputs.

No fitting, model downloads, dataset updates or segmentation jobs occur.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qa_ensemble.frozen import DeepSets, Jaccard, ensemble_matrix, relative_features, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-truth-root", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_path = args.silver_truth_root / "src/silver_truth/qa/cnn.py"
    tree = ast.parse(original_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Jaccard")
    namespace = {"torch": torch, "nn": torch.nn}
    for name in ("resnet18", "resnet50", "resnet101", "efficientnet_b1", "efficientnet_b4", "efficientnet_b7"):
        constructor = getattr(models, name)
        namespace[name] = lambda weights=None, c=constructor: c(weights=None)
    for name in ("ResNet18", "ResNet50", "ResNet101", "EfficientNet_B1", "EfficientNet_B4", "EfficientNet_B7"):
        namespace[name+"_Weights"] = type("Weights", (), {"IMAGENET1K_V1": None})
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(original_path), "exec"), namespace)
    saved = torch.load(args.qa, map_location="cpu", weights_only=True)
    metadata = saved.get("metadata", {})
    model_type = metadata.get("model_type", "resnet50")
    activation = metadata.get("output_activation", "linear")
    dropout = metadata.get("dropout_rate", .3)
    original = namespace["Jaccard"](model_type=model_type, dropout_rate=dropout, output_activation=activation).eval()
    adapter = Jaccard(model_type, dropout, activation).eval()
    original.load_state_dict(saved["model_state_dict"]); adapter.load_state_dict(saved["model_state_dict"])
    # Synthetic input compares implementations, not biological mask quality.
    tensor = torch.from_numpy(np.random.default_rng(13).integers(0, 256, size=(2, 2, 64, 64)).astype(np.float32)).div(255).sub(.5).div(.5)
    with torch.inference_mode():
        qa_error = float((original(tensor)-adapter(tensor)).abs().max())
    payload = joblib.load(args.ensemble)
    ensemble = DeepSets(payload["input_dim"], payload["config"]["deep_sets_hidden_dim"]).eval()
    ensemble.load_state_dict(payload["state_dict"])
    features = pd.read_parquet(args.candidate_table)
    max_error, count, groups = 0., 0, 0
    for _, group in features.groupby("ranker_cluster_id", sort=True):
        if groups == 100:
            break
        augmented = relative_features(group, payload)
        with torch.inference_mode():
            actual = ensemble(torch.from_numpy(ensemble_matrix(augmented, payload))).numpy()
        expected = group["candidate_set_weight_raw_score"].to_numpy()
        max_error = max(max_error, float(np.max(np.abs(actual-expected))))
        count += len(group); groups += 1
    if groups == 0:
        raise ValueError("No saved candidate groups to compare")
    report = {"qa_max_abs_error": qa_error, "ensemble_max_abs_error": max_error,
              "ensemble_candidate_rows": count, "ensemble_candidate_groups": groups,
              "qa_activation": activation, "qa_activation_source": "checkpoint metadata or original load_model compatibility default",
              "qa_sha256": sha256(args.qa), "ensemble_sha256": sha256(args.ensemble),
              "scope": "Component implementation parity only; ranker parity and full Labkit transfer not verified"}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if qa_error > 1e-6 or max_error > 1e-6:
        raise SystemExit("Frozen inference parity failed")


if __name__ == "__main__":
    main()
