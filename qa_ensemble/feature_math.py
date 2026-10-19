"""Inference math extracted unchanged from David Ciz's silver-truth code.
See vendor_provenance.json for source hashes and function names. No fitting code.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import ndimage
def _expanded_mask_rectangle(
    mask: np.ndarray,
    *,
    min_margin: int = 16,
    margin_fraction: float = 0.25,
) -> tuple[int, int, int, int] | None:
    area = float(mask.sum())
    if area <= 0:
        return None

    rows, cols = np.nonzero(mask)
    bbox_height = int(rows.max() - rows.min() + 1)
    bbox_width = int(cols.max() - cols.min() + 1)
    margin = int(np.ceil(max(float(min_margin), margin_fraction * max(bbox_height, bbox_width))))
    y0 = max(0, int(rows.min()) - margin)
    y1 = min(mask.shape[0], int(rows.max()) + margin + 1)
    x0 = max(0, int(cols.min()) - margin)
    x1 = min(mask.shape[1], int(cols.max()) + margin + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1

def _mask_raw_feature_record(
    raw: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
    include_full_boundary_noise: bool = False,
) -> dict[str, float]:
    area = float(mask.sum())
    area_fraction = float(area / mask.size) if mask.size > 0 else np.nan
    if area <= 0:
        return {
            f"{prefix}mask_area_pixels": 0.0,
            f"{prefix}mask_area_fraction": 0.0,
            f"{prefix}mask_component_count": 0.0,
        }

    rows, cols = np.nonzero(mask)
    bbox_height = float(rows.max() - rows.min() + 1)
    bbox_width = float(cols.max() - cols.min() + 1)
    bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else np.nan
    bbox_area = bbox_height * bbox_width
    extent = float(area / bbox_area) if bbox_area > 0 else np.nan

    eroded = ndimage.binary_erosion(mask)
    perimeter = float(np.logical_and(mask, ~eroded).sum())
    labeled, component_count = ndimage.label(mask)
    del labeled

    border_pixels = (
        float(mask[0, :].sum())
        + float(mask[-1, :].sum())
        + float(mask[:, 0].sum())
        + float(mask[:, -1].sum())
    )
    border_touch_fraction = border_pixels / max(perimeter, 1.0)

    centroid_y = float(rows.mean())
    centroid_x = float(cols.mean())
    center_y = (mask.shape[0] - 1) / 2.0
    center_x = (mask.shape[1] - 1) / 2.0
    diagonal = float(np.hypot(mask.shape[0], mask.shape[1]))
    centroid_offset = float(np.hypot(centroid_y - center_y, centroid_x - center_x))
    centroid_offset_norm = centroid_offset / diagonal if diagonal > 0 else np.nan

    dilated = ndimage.binary_dilation(mask, iterations=3)
    boundary_ring = np.logical_and(dilated, ~mask)
    inside = raw[mask]
    background = raw[boundary_ring]
    raw_inside_mean = _safe_mean(inside)
    raw_background_mean = _safe_mean(background)
    raw_background_std = _safe_std(background)
    raw_contrast = raw_inside_mean - raw_background_mean
    raw_contrast_norm = raw_contrast / raw_background_std if raw_background_std > 0 else np.nan

    boundary = np.logical_and(mask, ~eroded)
    raw_boundary_mean = _safe_mean(raw[boundary])
    raw_boundary_contrast = raw_boundary_mean - raw_background_mean

    record = {
        f"{prefix}mask_area_pixels": area,
        f"{prefix}mask_area_fraction": area_fraction,
        f"{prefix}mask_bbox_height": bbox_height,
        f"{prefix}mask_bbox_width": bbox_width,
        f"{prefix}mask_bbox_area": bbox_area,
        f"{prefix}mask_extent": extent,
        f"{prefix}mask_perimeter_pixels": perimeter,
        f"{prefix}mask_perimeter_area_ratio": perimeter / area,
        f"{prefix}mask_component_count": float(component_count),
        f"{prefix}mask_border_touch_fraction": border_touch_fraction,
        f"{prefix}mask_centroid_offset_norm": centroid_offset_norm,
        f"{prefix}raw_inside_mean": raw_inside_mean,
        f"{prefix}raw_local_background_mean": raw_background_mean,
        f"{prefix}raw_contrast": raw_contrast,
        f"{prefix}raw_contrast_norm": raw_contrast_norm,
        f"{prefix}raw_boundary_mean": raw_boundary_mean,
        f"{prefix}raw_boundary_contrast": raw_boundary_contrast,
    }
    if prefix:
        record[f"{prefix}mask_bbox_aspect_ratio"] = bbox_aspect_ratio
    if prefix == "full_" and include_full_boundary_noise:
        record.update(
            _full_image_boundary_noise_feature_record(
                raw=raw,
                mask=mask,
                boundary=boundary,
                boundary_ring=boundary_ring,
                inside=inside,
                background=background,
                inside_mean=raw_inside_mean,
                background_mean=raw_background_mean,
                background_std=raw_background_std,
                rows=rows,
                cols=cols,
                area=area,
            )
        )
    return record

def _full_image_boundary_noise_feature_record(
    *,
    raw: np.ndarray,
    mask: np.ndarray,
    boundary: np.ndarray,
    boundary_ring: np.ndarray,
    inside: np.ndarray,
    background: np.ndarray,
    inside_mean: float,
    background_mean: float,
    background_std: float,
    rows: np.ndarray,
    cols: np.ndarray,
    area: float,
) -> dict[str, float]:
    return {
        "full_raw_background_std": background_std,
        "full_raw_inside_std": _safe_std(inside),
        "full_raw_contrast_to_noise": (
            (inside_mean - background_mean) / (background_std + 1e-6)
            if np.isfinite(inside_mean) and np.isfinite(background_mean)
            else np.nan
        ),
        "full_raw_inside_p25": _safe_percentile(inside, 25),
        "full_raw_inside_p75": _safe_percentile(inside, 75),
        "full_raw_background_p25": _safe_percentile(background, 25),
        "full_raw_background_p75": _safe_percentile(background, 75),
        "full_raw_boundary_gradient_mean": _boundary_gradient_mean(
            raw, boundary
        ),
        "full_raw_boundary_gradient_contrast": _boundary_gradient_contrast(
            raw, boundary, boundary_ring
        ),
        "full_mask_hole_fraction": _mask_hole_fraction(mask, area=area),
        "full_mask_eccentricity": _mask_eccentricity(rows, cols),
    }

def _crop_geometry_feature_record(
    row: pd.Series, *, image_shape: tuple[int, int] | None
) -> dict[str, float]:
    values = [
        row.get("crop_y_start"),
        row.get("crop_y_end"),
        row.get("crop_x_start"),
        row.get("crop_x_end"),
    ]
    if image_shape is None or any(pd.isna(value) for value in values):
        return {}

    y_start, y_end, x_start, x_end = [float(value) for value in values]
    image_height, image_width = image_shape
    crop_height = max(0.0, y_end - y_start)
    crop_width = max(0.0, x_end - x_start)
    crop_area = crop_height * crop_width

    overflow_top = max(0.0, -y_start)
    overflow_bottom = max(0.0, y_end - image_height)
    overflow_left = max(0.0, -x_start)
    overflow_right = max(0.0, x_end - image_width)
    visible_height = max(0.0, min(y_end, image_height) - max(y_start, 0.0))
    visible_width = max(0.0, min(x_end, image_width) - max(x_start, 0.0))
    visible_fraction = (
        (visible_height * visible_width) / crop_area if crop_area > 0 else np.nan
    )

    return {
        "crop_overflow_top": overflow_top,
        "crop_overflow_bottom": overflow_bottom,
        "crop_overflow_left": overflow_left,
        "crop_overflow_right": overflow_right,
        "crop_visible_height": visible_height,
        "crop_visible_width": visible_width,
        "crop_visible_fraction": visible_fraction,
        "crop_has_padding": float(
            overflow_top > 0
            or overflow_bottom > 0
            or overflow_left > 0
            or overflow_right > 0
        ),
    }

def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))

def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values))

def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))

def _gradient_magnitude(raw: np.ndarray) -> np.ndarray | None:
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
        return None
    grad_y, grad_x = np.gradient(raw.astype(float, copy=False))
    return np.hypot(grad_y, grad_x)

def _boundary_gradient_mean(raw: np.ndarray, boundary: np.ndarray) -> float:
    gradient = _gradient_magnitude(raw)
    if gradient is None:
        return float("nan")
    return _safe_mean(gradient[boundary])

def _boundary_gradient_contrast(
    raw: np.ndarray, boundary: np.ndarray, boundary_ring: np.ndarray
) -> float:
    gradient = _gradient_magnitude(raw)
    if gradient is None:
        return float("nan")
    return _safe_mean(gradient[boundary]) - _safe_mean(gradient[boundary_ring])

def _mask_hole_fraction(mask: np.ndarray, *, area: float) -> float:
    if area <= 0:
        return 0.0
    filled = ndimage.binary_fill_holes(mask)
    hole_area = float(filled.sum()) - area
    return hole_area / max(area, 1.0)

def _mask_eccentricity(rows: np.ndarray, cols: np.ndarray) -> float:
    if rows.size < 3:
        return float("nan")
    coords = np.vstack([rows.astype(float), cols.astype(float)])
    covariance = np.cov(coords)
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(np.max(eigenvalues))
    smallest = float(np.min(eigenvalues))
    if largest <= 0:
        return float("nan")
    return float(np.sqrt(max(0.0, 1.0 - (smallest / largest))))

def add_cluster_ranker_features(
    features: pd.DataFrame, *, cluster_distance: float = 4.0
) -> pd.DataFrame:
    if not {"image_id", "center_y", "center_x"}.issubset(features.columns):
        return features.copy()

    augmented = features.copy()
    augmented["ranker_cluster_id"] = _assign_center_clusters(
        augmented, max_distance=cluster_distance
    )
    augmented["ranker_cluster_size"] = augmented.groupby(
        "ranker_cluster_id", dropna=False
    )["ranker_cluster_id"].transform("size")
    augmented["ranker_cluster_qa_rank"] = _rank_within_cluster(augmented, "qa_score")
    augmented["ranker_cluster_source_prior_rank"] = _rank_within_cluster(
        augmented, "source_train_mean_iou"
    )
    source_temporal_distance = _numeric_series(
        augmented, "source_temporal_min_center_dist_norm"
    )
    any_temporal_distance = _numeric_series(augmented, "temporal_min_center_dist_norm")
    temporal_distance = source_temporal_distance.fillna(any_temporal_distance)
    source_temporal_support = _numeric_series(
        augmented, "source_temporal_support_score"
    )
    any_temporal_support = _numeric_series(augmented, "temporal_support_score")
    temporal_support = source_temporal_support.fillna(any_temporal_support)
    augmented["_ranker_temporal_score"] = (
        1.0 / (1.0 + temporal_distance)
    ).clip(0.0, 1.0)
    augmented["_ranker_temporal_support_score"] = temporal_support.fillna(
        augmented["_ranker_temporal_score"]
    ).clip(0.0, 1.0)
    augmented["ranker_cluster_temporal_rank"] = augmented.groupby(
        "ranker_cluster_id", dropna=False
    )["_ranker_temporal_score"].rank(method="average", pct=True)
    augmented["ranker_cluster_temporal_support_rank"] = augmented.groupby(
        "ranker_cluster_id", dropna=False
    )["_ranker_temporal_support_score"].rank(method="average", pct=True)
    _add_image_cluster_relative_features(augmented)
    return augmented.drop(
        columns=["_ranker_temporal_score", "_ranker_temporal_support_score"]
    )

def _add_image_cluster_relative_features(features: pd.DataFrame) -> None:
    if "ranker_cluster_id" not in features.columns:
        return

    _add_prefixed_cluster_relative_features(features, prefix="rect")
    _add_prefixed_cluster_relative_features(features, prefix="full")

def _add_prefixed_cluster_relative_features(
    features: pd.DataFrame, *, prefix: str
) -> None:
    contrast_column = f"{prefix}_raw_contrast_norm"
    boundary_column = f"{prefix}_raw_boundary_contrast"
    extent_column = f"{prefix}_mask_extent"
    area_column = f"{prefix}_mask_area_pixels"

    if contrast_column in features.columns:
        features[f"ranker_cluster_{prefix}_raw_contrast_norm_rank"] = (
            _rank_within_cluster(features, contrast_column)
        )
    if boundary_column in features.columns:
        features[f"ranker_cluster_{prefix}_boundary_contrast_rank"] = (
            _rank_within_cluster(features, boundary_column)
        )
    if extent_column in features.columns:
        features[f"ranker_cluster_{prefix}_mask_extent_rank"] = _rank_within_cluster(
            features, extent_column
        )
    if area_column in features.columns:
        area = _numeric_series(features, area_column)
        median_area = area.groupby(features["ranker_cluster_id"], dropna=False).transform(
            "median"
        )
        ratio = area / median_area.replace(0, np.nan)
        log_ratio = np.log(ratio.replace(0, np.nan))
        features[f"{prefix}_mask_area_log_ratio_to_cluster_median"] = log_ratio
        features[f"{prefix}_mask_area_abs_log_ratio_to_cluster_median"] = np.abs(
            log_ratio
        )

def _rank_within_cluster(features: pd.DataFrame, column: str) -> pd.Series:
    if column not in features.columns:
        return pd.Series(np.nan, index=features.index, dtype=float)
    return features.groupby("ranker_cluster_id", dropna=False)[column].rank(
        method="average", pct=True
    )

def _numeric_series(features: pd.DataFrame, column: str) -> pd.Series:
    if column not in features.columns:
        return pd.Series(np.nan, index=features.index, dtype=float)
    return pd.to_numeric(features[column], errors="coerce")

def _assign_center_clusters(
    features: pd.DataFrame, *, max_distance: float
) -> pd.Series:
    cluster_ids = pd.Series(index=features.index, dtype=object)
    for image_id, group in features.groupby("image_id", dropna=False, sort=False):
        centers = group[["center_y", "center_x"]].to_numpy(dtype=float)
        labels = _connected_components(centers, max_distance=max_distance)
        cluster_ids.loc[group.index] = [
            f"{image_id}__cluster_{label:04d}" for label in labels
        ]
    return cluster_ids

def _connected_components(points: np.ndarray, *, max_distance: float) -> list[int]:
    n_points = len(points)
    parent = list(range(n_points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(n_points):
        if np.isnan(points[left]).any():
            continue
        for right in range(left + 1, n_points):
            if np.isnan(points[right]).any():
                continue
            if float(np.linalg.norm(points[left] - points[right])) <= max_distance:
                union(left, right)

    roots: dict[int, int] = {}
    labels = []
    for index in range(n_points):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels

def build_pairwise_scoring_table(
    group: pd.DataFrame,
    *,
    feature_columns: list[str],
    use_source_identity: bool,
) -> tuple[pd.DataFrame, list[int]]:
    records = []
    left_indices = []
    rows = list(group.iterrows())
    for left_index, left in rows:
        for right_index, right in rows:
            if left_index == right_index:
                continue
            records.append(
                _pair_feature_record(
                    left,
                    right,
                    feature_columns=feature_columns,
                    use_source_identity=use_source_identity,
                )
            )
            left_indices.append(left_index)
    return pd.DataFrame.from_records(records), left_indices

def _pair_feature_record(
    left: pd.Series,
    right: pd.Series,
    *,
    feature_columns: list[str],
    use_source_identity: bool,
) -> dict[str, object]:
    record: dict[str, object] = {}
    for column in feature_columns:
        left_value = pd.to_numeric(pd.Series([left.get(column)]), errors="coerce").iloc[
            0
        ]
        right_value = pd.to_numeric(
            pd.Series([right.get(column)]), errors="coerce"
        ).iloc[0]
        record[f"diff__{column}"] = left_value - right_value
    if use_source_identity:
        record["left_source"] = str(left.get("candidate_source", ""))
        record["right_source"] = str(right.get("candidate_source", ""))
    return record
