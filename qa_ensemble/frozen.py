"""Inference-only adapters for the existing silver-truth artifacts. No training."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage
import torch
from torch import nn
from torchvision import models

from . import feature_math as fm

logger = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Jaccard(nn.Module):
    """Same state-dict layout as silver_truth.qa.cnn.Jaccard; no weight download."""
    def __init__(self, model_type, dropout_rate, output_activation):
        super().__init__()
        if model_type not in {"resnet18", "resnet50", "resnet101", "efficientnet_b1", "efficientnet_b4", "efficientnet_b7"}:
            raise ValueError("Unsupported QA architecture")
        if output_activation not in {"linear", "sigmoid"}:
            raise ValueError("Unsupported QA activation")
        self.output_activation = output_activation
        self.model = getattr(models, model_type)(weights=None)
        if model_type.startswith("resnet"):
            width = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.model.conv1 = nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False)
        else:
            width = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
            first = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(2, first.out_channels, 3, stride=2, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(width, 1)

    def forward(self, x):
        x = self.fc(self.dropout(self.model(x)))
        return torch.sigmoid(x) if self.output_activation == "sigmoid" else x


class DeepSets(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        encoded = self.phi(x)
        context = encoded.mean(dim=0, keepdim=True).expand_as(encoded)
        return self.rho(torch.cat([encoded, context], dim=1)).squeeze(1)


def relative_features(features, payload):
    version = payload.get("candidate_grouping_version", 1)
    if version not in (1, 2):
        raise ValueError(f"Unsupported candidate-grouping version: {version}")
    if version == 2:
        stale = [column for column in features if column.startswith((
            "candidate_set_rank__", "candidate_set_centered__",
            "candidate_set_gap_to_best__", "candidate_set_context__"))]
        work = features.drop(columns=stale).copy()
        keys = [column for column in payload["config"].get("group_context_columns", (
            "dataset", "fold", "campaign_number", "candidate_set_id", "label")) if column in work]
        keys += [column for column in ("split", "image_id") if column in work]
        namespace = (work[keys].astype("string").fillna("<missing>").apply(
            lambda row: json.dumps(list(zip(keys, row.tolist()))), axis=1)
            if keys else pd.Series("all_candidates", index=work.index))
        original_image = work["image_id"].copy() if "image_id" in work else None
        work["image_id"] = namespace
        augmented = fm.add_cluster_ranker_features(
            work, cluster_distance=payload["config"].get("cluster_distance", 4.0))
        if not {"center_y", "center_x"}.issubset(work.columns):
            augmented["ranker_cluster_id"] = namespace
        if original_image is None:
            augmented = augmented.drop(columns="image_id")
        else:
            augmented["image_id"] = original_image
    else:
        # Keep the original bundle's inference unchanged for explicit rollback.
        augmented = fm.add_cluster_ranker_features(features, cluster_distance=payload["config"].get("cluster_distance", 4.0))
    additions = {}
    for column in payload["config"]["feature_columns"]:
        if column not in augmented:
            continue
        values = pd.to_numeric(augmented[column], errors="coerce")
        grouped = values.groupby(augmented["ranker_cluster_id"], dropna=False)
        additions[f"candidate_set_rank__{column}"] = grouped.rank(method="average", pct=True)
        additions[f"candidate_set_centered__{column}"] = values - grouped.transform("mean")
        additions[f"candidate_set_gap_to_best__{column}"] = grouped.transform("max") - values
    return pd.concat([augmented, pd.DataFrame(additions, index=augmented.index)], axis=1)


def ensemble_matrix(features, payload):
    columns = payload["feature_columns"]
    # The saved tables can already contain derived columns. The original loader
    # consistently takes the first occurrence when a column is duplicated.
    numeric = pd.DataFrame({column: pd.to_numeric(features[column].iloc[:, 0] if isinstance(features[column], pd.DataFrame) else features[column], errors="coerce") for column in columns}, index=features.index)
    means = pd.Series(payload["preprocess"]["numeric_mean"])
    stds = pd.Series(payload["preprocess"]["numeric_std"])
    numeric = ((numeric.fillna(means) - means) / stds).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    parts = [numeric.to_numpy(dtype=np.float32)]
    for column in payload["categorical_columns"]:
        values = features[column].fillna("").astype(str)
        categories = payload["preprocess"]["categories"].get(column, [])
        parts.append(np.array([[float(value == category) for category in categories] for value in values], dtype=np.float32).reshape(len(values), len(categories)))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def ensemble_raw_scores(features, payload, model, device="cpu"):
    """Evaluate each candidate set independently, including multi-cell replays."""
    matrix = ensemble_matrix(features, payload)
    output = np.zeros(len(features), dtype=float)
    for indices in features.groupby("ranker_cluster_id", dropna=False, sort=False).indices.values():
        with torch.inference_mode():
            output[indices] = model(torch.as_tensor(matrix[indices], device=device)).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Ensemble produced nonfinite scores")
    return output


def centered_crop(array, center, size):
    cx, cy = center
    x0, y0 = cx - size // 2, cy - size // 2
    h, w = array.shape
    x1, y1 = x0 + size, y0 + size
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    cropped = np.zeros((size, size), dtype=array.dtype)
    cropped[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = array[sy0:sy1, sx0:sx1]
    return cropped, (x0, y0, x1, y1)


def paste_crop(array, box, shape):
    x0, y0, x1, y1 = box
    h, w = shape
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    result = np.zeros(shape, dtype=array.dtype)
    result[sy0:sy1, sx0:sx1] = array[sy0-y0:sy1-y0, sx0-x0:sx1-x0]
    return result


def make_features(raw, crop_raw, crop_masks, roi_masks, scores, request, box, bundle):
    records = []
    cx, cy = request.anchor
    ox, oy = request.region.origin_xy
    original_shape = request.region.original_shape_yx
    roi_is_full = (ox, oy) == (0, 0) and tuple(raw.shape) == tuple(original_shape)
    for method, mask in crop_masks.items():
        record = {"image_id": "request", "center_x": cx, "center_y": cy,
                  "candidate_source": method, "dataset": bundle["dataset_identity"], "qa_score": scores[method],
                  "crop_width": mask.shape[1], "crop_height": mask.shape[0], "crop_area": mask.size,
                  "fusion_support": float(len(crop_masks)),
                  "source_train_mean_iou": np.nan, "dataset_train_mean_iou": np.nan,
                  "source_temporal_min_center_dist_norm": np.nan, "temporal_min_center_dist_norm": np.nan,
                  "source_temporal_support_score": np.nan, "temporal_support_score": np.nan}
        record.update(fm._mask_raw_feature_record(crop_raw, mask, prefix=""))
        rect = fm._expanded_mask_rectangle(mask)
        if rect is not None:
            y0, y1, x0, x1 = rect
            record.update(fm._mask_raw_feature_record(crop_raw[y0:y1, x0:x1], mask[y0:y1, x0:x1], prefix="rect_"))
        # Full-context features require all of the original expanded rectangle.
        roi_mask = paste_crop(mask, box, raw.shape)
        rect = fm._expanded_mask_rectangle(roi_mask)
        if rect is not None:
            y0, y1, x0, x1 = rect
            rows, cols = np.nonzero(roi_mask)
            margin = int(np.ceil(max(16, .25 * max(np.ptp(rows)+1, np.ptp(cols)+1))))
            supplied_context = (rows.min() >= margin and cols.min() >= margin and
                                rows.max()+margin < raw.shape[0] and cols.max()+margin < raw.shape[1])
            if roi_is_full or supplied_context:
                record.update(fm._mask_raw_feature_record(raw[y0:y1, x0:x1], roi_mask[y0:y1, x0:x1], prefix="full_", include_full_boundary_noise=True))
                record["full_mask_retained_fraction"] = float(roi_mask.sum() / max(1, mask.sum()))
        crop_x0, crop_y0, crop_x1, crop_y1 = box
        record.update(fm._crop_geometry_feature_record(pd.Series({"crop_x_start": crop_x0+ox,
                      "crop_x_end": crop_x1+ox, "crop_y_start": crop_y0+oy, "crop_y_end": crop_y1+oy}), image_shape=original_shape))
        record["mask_touching_square_crop_border"] = float(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        rectangle = fm._expanded_mask_rectangle(mask)
        if rectangle is not None:
            ry0, ry1, rx0, rx1 = rectangle
            rectangle_mask = mask[ry0:ry1, rx0:rx1]
            record["mask_touching_rect_border"] = float(rectangle_mask[0].any() or rectangle_mask[-1].any() or rectangle_mask[:, 0].any() or rectangle_mask[:, -1].any())
        rows, cols = np.nonzero(mask)
        if len(rows):
            record["mask_touching_image_border"] = float((rows+crop_y0+oy).min() <= 0 or (rows+crop_y0+oy).max() >= original_shape[0]-1 or (cols+crop_x0+ox).min() <= 0 or (cols+crop_x0+ox).max() >= original_shape[1]-1)
        records.append(record)
    features = pd.DataFrame(records)
    # Whole-image population ranks are unavailable in a cell-by-cell request.
    # Local candidate ranks are still computed by add_cluster_ranker_features.
    features["fusion_support_rank_in_image"] = np.nan
    features["qa_rank_in_image"] = np.nan
    return fm.add_cluster_ranker_features(features)


def rank_candidates(features, payload):
    table = features.copy()
    for column in payload["feature_columns"]:
        if column not in table:
            table[column] = np.nan
    if len(table) == 1:
        probabilities = np.array([.5])
    else:
        pairs, indices = fm.build_pairwise_scoring_table(table, feature_columns=payload["feature_columns"], use_source_identity=payload["config"].get("use_source_identity", False))
        probability = payload["model"].predict_proba(pairs[payload["model_columns"]])[:, 1]
        if payload.get("source_agnostic_model") is not None:
            blend = payload["config"].get("source_identity_blend", 1.)
            probability = blend * probability + (1-blend) * payload["source_agnostic_model"].predict_proba(pairs[payload["numeric_columns"]])[:, 1]
        probabilities = pd.DataFrame({"index": indices, "prob": probability}).groupby("index")["prob"].mean().reindex(table.index).to_numpy()
    table["pairwise_cluster_score"] = np.clip(probabilities, 0., 1.)
    table["pairwise_cluster_rank"] = table["pairwise_cluster_score"].rank(method="average", pct=True)
    return table


class FrozenPipeline:
    def __init__(self, manifest_path, device="cpu"):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if m["schema_version"] != 1 or m["ensemble"]["weight_transform"] != "raw_softmax":
            raise ValueError("Unsupported frozen pipeline schema or weighting")
        settings = m["ensemble"]
        if (not isinstance(settings["max_candidates"], int) or not 1 <= settings["max_candidates"] <= 16
                or not np.isfinite(settings["temperature"]) or settings["temperature"] <= 0
                or not 0 <= settings["threshold"] <= 1 or not 0 <= settings["boundary_floor"] <= 1):
            raise ValueError("Invalid frozen selection or voting settings")
        self.device = torch.device(device)
        def artifact(key):
            info = m["artifacts"][key]
            path = (self.path.parent / info["file"]).resolve()
            if self.path.parent.resolve() not in path.parents or sha256(path) != info["sha256"]:
                raise ValueError(f"Invalid or modified {key} artifact")
            return path
        qa = torch.load(artifact("qa"), map_location="cpu", weights_only=True)
        cfg = m["qa"]
        if cfg["raw_encoding"] not in {"preserve", "minmax_uint8"} or cfg["tensor_transform"] != "divide255_center_half":
            raise ValueError("Unsupported QA preprocessing profile")
        if not isinstance(cfg["crop_size"], int) or cfg["crop_size"] < 16 or cfg["crop_size"] > 2048:
            raise ValueError("QA crop size must be between 16 and 2048 pixels")
        self.qa = Jaccard(cfg["model_type"], cfg["dropout_rate"], cfg["output_activation"])
        self.qa.load_state_dict(qa["model_state_dict"])
        self.qa.to(self.device).eval()
        # Only trusted, hash-checked, server-installed sklearn payloads are loaded.
        self.ranker = joblib.load(artifact("ranker"))
        self.payload = torch.load(artifact("ensemble"), map_location="cpu", weights_only=True)
        version = self.payload.get("candidate_grouping_version", 1)
        if version not in (1, 2) or version != settings.get("candidate_grouping_version", 1):
            raise ValueError("Bundle and ensemble candidate-grouping versions differ or are unsupported")
        self.ensemble = DeepSets(self.payload["input_dim"], self.payload["config"]["deep_sets_hidden_dim"])
        self.ensemble.load_state_dict(self.payload["state_dict"])
        self.ensemble.to(self.device).eval()
        self.provenance = {"pipeline_id": m["pipeline_id"], "manifest_sha256": sha256(self.path),
                           "artifacts": m["artifacts"], "training_exposure": m["training_exposure"],
                           "qa_profile": m["qa"], "ensemble_settings": m["ensemble"],
                           "candidate_grouping_version": version}

    def run(self, image, candidates, request):
        result = {"scores": [], "stages": {}, "diagnostics": {}}
        stage = "qa"
        try:
            cfg = self.manifest["qa"]
            raw = image.astype(np.float32)
            if cfg["raw_encoding"] == "minmax_uint8":
                span = float(raw.max()-raw.min())
                raw = np.zeros_like(raw) if span == 0 else ((raw-raw.min()) / span * 255).astype(np.uint8)
            crop_raw, box = centered_crop(raw, request.anchor, cfg["crop_size"])
            crop_masks = {key: centered_crop(mask, request.anchor, cfg["crop_size"])[0] for key, mask in candidates.items()}
            inputs = np.stack([np.stack([crop_raw, mask.astype(np.float32)*255]) for mask in crop_masks.values()]).astype(np.float32)
            tensor = torch.as_tensor(inputs, device=self.device).div(255.).sub(.5).div(.5)
            with torch.inference_mode():
                predicted = self.qa(tensor).flatten().cpu().numpy()
            if not np.isfinite(predicted).all():
                raise ValueError("QA produced a nonfinite score")
            scores = dict(zip(crop_masks, map(float, predicted)))
            result["scores"] = [{"method": key, "qa_score": value} for key, value in scores.items()]
            result["candidate_masks"] = {key: paste_crop(mask, box, image.shape) for key, mask in crop_masks.items()}
            result["stages"][stage] = {"status": "ok"}
            result["diagnostics"]["qa_crop_box_xyxy"] = box
            result["diagnostics"]["qa_crop_padding"] = any((box[0] < 0, box[1] < 0, box[2] > image.shape[1], box[3] > image.shape[0]))
            result["diagnostics"]["candidate_crop_retained_fraction"] = {key: float(mask.sum()/max(1, candidates[key].sum())) for key, mask in crop_masks.items()}
            if any(value < 1. for value in result["diagnostics"]["candidate_crop_retained_fraction"].values()):
                result["warnings"] = ["Some candidate masks extend beyond the frozen QA crop. The returned ensemble is limited to that crop; inspect it before applying."]

            stage = "ranking"
            features = make_features(raw, crop_raw, crop_masks, candidates, scores, request, box, self.manifest)
            features = rank_candidates(features, self.ranker)
            # Same local support recipe as the preserved paper-facing inference.
            support_columns = ["ranker_cluster_full_boundary_contrast_rank", "ranker_cluster_rect_boundary_contrast_rank",
                               "ranker_cluster_full_raw_contrast_norm_rank", "ranker_cluster_rect_raw_contrast_norm_rank"]
            features["local_image_support_score"] = features.reindex(columns=support_columns).mean(axis=1).fillna(.5).clip(0, 1)
            count = self.manifest["ensemble"]["max_candidates"]
            selected = features.sort_values("pairwise_cluster_score", ascending=False, kind="stable").head(count).copy()
            chosen = set(selected["candidate_source"])
            for row, score in zip(result["scores"], features["pairwise_cluster_score"]):
                row.update(ranking_score=float(score), retained=row["method"] in chosen,
                           selection_reason="retained" if row["method"] in chosen else "outside_top_k", fusion_weight=0.)
            result["stages"][stage] = {"status": "ok", "retained_count": len(selected)}

            stage = "ensemble"
            for column in self.payload["config"]["feature_columns"]:
                if column not in selected:
                    selected[column] = np.nan
            selected = relative_features(selected, self.payload)
            for column in self.payload["feature_columns"]:
                if column not in selected:
                    selected[column] = np.nan
            result["diagnostics"]["imputed_ensemble_features"] = [column for column in self.payload["feature_columns"] if selected[column].isna().any()]
            result["diagnostics"]["unknown_categories"] = {column: sorted(set(selected[column].astype(str))-set(categories)) for column, categories in self.payload["preprocess"]["categories"].items()}
            logits = ensemble_raw_scores(selected, self.payload, self.ensemble, self.device)
            params = self.manifest["ensemble"]
            weights = np.exp(np.clip((logits-logits.max()) / params["temperature"], -50., 0.)).clip(min=1e-3)
            weights /= weights.sum()
            vote = np.zeros(crop_raw.shape, dtype=np.float32)
            for method, weight, support in zip(selected["candidate_source"], weights, selected["local_image_support_score"]):
                mask = crop_masks[method]
                boundary = mask & ~ndimage.binary_erosion(mask, iterations=1)
                support = np.clip(support, params["boundary_floor"], 1.)
                vote += mask * np.where(boundary, weight*support, weight).astype(np.float32)
                next(row for row in result["scores"] if row["method"] == method)["fusion_weight"] = float(weight)
            if not np.isfinite(vote).all():
                raise ValueError("Ensemble produced nonfinite vote values")
            result["vote_map"] = paste_crop(vote, box, image.shape)
            result["ensemble_mask"] = paste_crop(vote >= params["threshold"], box, image.shape)
            result["stages"][stage] = {"status": "ok", "mode": "single_candidate" if len(selected) == 1 else "fusion"}
        except Exception as exc:
            logger.exception("Frozen pipeline failed at %s", stage)
            result["stages"][stage] = {"status": "failed", "error": str(exc)}
        return result


def load_pipelines(root, device="cpu"):
    pipelines, failures = {}, []
    for manifest in sorted(Path(root).glob("*/manifest.json")):
        try:
            pipeline = FrozenPipeline(manifest, device)
            identity = pipeline.manifest["pipeline_id"]
            if identity in pipelines:
                raise ValueError("Duplicate pipeline ID")
            pipelines[identity] = pipeline
        except Exception as exc:
            logger.warning("Cannot load frozen pipeline %s: %s", manifest.parent.name, exc)
            failures.append({"bundle": manifest.parent.name, "status": "unavailable", "reason": str(exc)})
    return pipelines, failures
