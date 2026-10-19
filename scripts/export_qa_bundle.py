#!/usr/bin/env python3
"""Export already-trained, trusted artifacts into a frozen server bundle.

This command never fits weights. --profile records explicitly chosen preprocessing
and final fusion settings; it must come from the matching original experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--ranker", required=True, type=Path)
    parser.add_argument("--ensemble", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    for key in ("pipeline_id", "qa", "ensemble", "training_exposure", "dataset_identity"):
        if key not in profile:
            parser.error(f"Profile is missing {key}")
    if args.output.exists():
        parser.error("Use a new output directory to avoid replacing an existing bundle")
    qa = torch.load(args.qa, map_location="cpu", weights_only=True)
    if "model_state_dict" not in qa:
        parser.error("Expected a QA checkpoint saved by silver-truth save_model")
    # joblib is for trusted local artifacts only, never an upload from an API user.
    ensemble = joblib.load(args.ensemble)
    if ensemble.get("model_type") != "deep_sets":
        parser.error("Expected a saved Deep Sets payload")
    ranker = joblib.load(args.ranker)
    if not {"model", "feature_columns", "model_columns", "config"}.issubset(ranker):
        parser.error("Expected a saved pairwise ranker payload")
    args.output.mkdir(parents=True)
    torch.save(qa, args.output / "qa.pt")
    torch.save(ensemble, args.output / "ensemble.pt")
    shutil.copy2(args.ranker, args.output / "ranker.joblib")
    profile["schema_version"] = 1
    profile["artifacts"] = {}
    for key, filename in (("qa", "qa.pt"), ("ranker", "ranker.joblib"), ("ensemble", "ensemble.pt")):
        profile["artifacts"][key] = {"file": filename, "sha256": hashlib.sha256((args.output / filename).read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(f"Exported frozen bundle to {args.output}")


if __name__ == "__main__":
    main()
