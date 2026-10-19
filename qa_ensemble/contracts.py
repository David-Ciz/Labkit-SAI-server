from __future__ import annotations

from io import BytesIO
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
import tifffile

OUTPUTS = ("ensemble_mask", "source_segmentations", "candidate_masks", "qa_scores",
           "selection", "ensemble_weights", "vote_map", "diagnostics")


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin_xy: tuple[StrictInt, StrictInt] = (0, 0)
    original_shape_yx: tuple[StrictInt, StrictInt]
    target_box_xyxy: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    target_point_xy: tuple[StrictInt, StrictInt] | None = None


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_version: Literal[1] = 1
    image_id: str = Field(default="image", min_length=1, max_length=256)
    coordinate_space: Literal["source_image", "captured_view"] = "source_image"
    view_to_source_affine: tuple[float, float, float, float, float, float, float, float, float, float, float, float] | None = None
    region: Region
    methods: list[str] = Field(min_length=1, max_length=16)
    pipeline_id: str = Field(min_length=1, max_length=128)
    outputs: list[Literal["ensemble_mask", "source_segmentations", "candidate_masks",
                          "qa_scores", "selection", "ensemble_weights", "vote_map",
                          "diagnostics"]] = Field(default_factory=lambda: ["ensemble_mask", "qa_scores"], alias="return")

    @model_validator(mode="after")
    def unique_methods(self):
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Choose each segmentation method only once")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Choose each output only once")
        if self.view_to_source_affine is not None and not np.isfinite(self.view_to_source_affine).all():
            raise ValueError("Coordinate transform must contain only finite values")
        return self

    def validate_image(self, shape):
        h, w = shape
        ox, oy = self.region.origin_xy
        ih, iw = self.region.original_shape_yx
        x0, y0, x1, y1 = self.region.target_box_xyxy
        if min(ox, oy) < 0 or min(ih, iw) <= 0 or ox + w > iw or oy + h > ih:
            raise ValueError("The context ROI does not fit in the original image")
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Target box must be a nonempty half-open box inside the ROI")
        px, py = self.anchor
        if not (x0 <= px < x1 and y0 <= py < y1):
            raise ValueError("Target point must lie inside the target box")

    @property
    def anchor(self):
        if self.region.target_point_xy is not None:
            return self.region.target_point_xy
        x0, y0, x1, y1 = self.region.target_box_xyxy
        return ((x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2)


def decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    """Check dimensions before decompressing TIFF pixel data."""
    try:
        with tifffile.TiffFile(BytesIO(data)) as tif:
            if len(tif.pages) != 1:
                raise ValueError("Upload one 2D grayscale TIFF page")
            page = tif.pages[0]
            if len(page.shape) != 2 or np.prod(page.shape, dtype=np.int64) > max_pixels:
                raise ValueError("Unsupported dimensions or ROI pixel limit exceeded")
            if page.dtype.kind not in "uif" or page.dtype.itemsize > 8:
                raise ValueError("Use an unsigned integer or real-valued grayscale image")
            image = page.asarray()
    except (tifffile.TiffFileError, OSError, IndexError) as exc:
        raise ValueError("Invalid TIFF image") from exc
    if image.size == 0 or not np.isfinite(image).all():
        raise ValueError("Image must be nonempty and contain only finite intensities")
    return image


def target_mask(labels, request: RunRequest):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype.kind not in "uib" or labels.min(initial=0) < 0 or labels.max(initial=0) > 2**32 - 1:
        raise ValueError("Segmenter must return a 2D unsigned integer label image")
    px, py = request.anchor
    label = int(labels[py, px])
    if label:
        return labels == label, {"target_label": label, "match": "anchor"}
    x0, y0, x1, y1 = request.region.target_box_xyxy
    ids = np.unique(labels[y0:y1, x0:x1])
    ids = ids[ids != 0]
    # A single non-background label in the target box is an unambiguous fallback.
    # Several labels require a better anchor; do not guess using their sizes.
    if len(ids) == 1:
        label = int(ids[0])
        return labels == label, {"target_label": label, "match": "single_instance_in_box"}
    if len(ids) == 0:
        raise ValueError("No segmented target in the selected box")
    raise ValueError("Ambiguous target: place the target point inside the intended cell")
