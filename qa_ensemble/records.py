"""Opt-in training records: what the pool proposed for one cell, and what the user kept.

Every interactive QA/ensemble request already produces all candidates for one target
cell. With LABKIT_RECORD_DIR set, the server stores them per run; the client later
posts the mask the user actually applied (``/segmentation_2D/qa_ensemble/feedback``).
``scripts/train_pool_bundle.py --records`` trains a pool-specific bundle from them.

Layout of one record (directory name = run_id):
    request.json      validated request (region in context-ROI coordinates)
    image.tif         the uploaded context ROI, original dtype
    candidates.json   [{"method", "file"}] for target-matched candidates
    candidates/NNN.tif uint8 masks, ROI coordinates
    scores.csv        QA/ranking/fusion table returned to the user (if any)
    final_mask.tif    uint8 mask the user applied (after feedback)
    feedback.json     {"action", "received_at"}

Records contain users' image data; enable only where storing it is permitted.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
import json
import logging
import os
from pathlib import Path
import re
import shutil

import numpy as np
import tifffile

logger = logging.getLogger(__name__)
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
ACTIONS = ("applied", "edited", "rejected")


def _write_mask(path: Path, mask):
    tifffile.imwrite(path, (np.asarray(mask) > 0).astype(np.uint8), photometric="minisblack", compression="zlib")


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls):
        root = os.getenv("LABKIT_RECORD_DIR", "").strip()
        return cls(root) if root else None

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id):
            raise ValueError("Invalid run_id")
        return self.root / run_id

    def save(self, run_id, image, request, candidates, scores):
        """Write atomically (temporary directory, then rename); never partially visible."""
        final = self._run_dir(run_id)
        tmp = self.root / f".tmp-{run_id}"
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            (tmp / "request.json").write_text(request.model_dump_json(by_alias=True))
            tifffile.imwrite(tmp / "image.tif", np.asarray(image), photometric="minisblack", compression="zlib")
            (tmp / "candidates").mkdir()
            index = []
            for i, (method, mask) in enumerate(candidates.items()):
                name = f"candidates/{i:03d}.tif"
                _write_mask(tmp / name, mask)
                index.append({"method": method, "file": name})
            (tmp / "candidates.json").write_text(json.dumps(index))
            if scores:
                text = StringIO()
                writer = csv.DictWriter(text, fieldnames=sorted({k for row in scores for k in row}))
                writer.writeheader()
                writer.writerows(scores)
                (tmp / "scores.csv").write_text(text.getvalue())
            tmp.rename(final)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def feedback(self, run_id, action, mask=None):
        run = self._run_dir(run_id)
        if not run.is_dir():
            raise FileNotFoundError("Unknown run_id")
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        if action != "rejected":
            if mask is None:
                raise ValueError("A mask is required unless the result was rejected")
            shape = tifffile.TiffFile(run / "image.tif").pages[0].shape
            mask = np.asarray(mask)
            if mask.shape != tuple(shape):
                raise ValueError(f"Mask shape {mask.shape} differs from the recorded ROI {tuple(shape)}")
            if not (mask > 0).any():
                raise ValueError("Mask is empty")
            _write_mask(run / "final_mask.tif", mask)
        else:
            (run / "final_mask.tif").unlink(missing_ok=True)
        (run / "feedback.json").write_text(json.dumps(
            {"action": action, "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))


def load_records(root):
    """Yield (run_id, request dict, image, final mask, {method: mask}) for records with a kept mask."""
    for run in sorted(Path(root).iterdir()):
        if not run.is_dir() or not RUN_ID.match(run.name) or not (run / "final_mask.tif").exists():
            continue
        request = json.loads((run / "request.json").read_text())
        candidates = {c["method"]: tifffile.imread(run / c["file"]) > 0
                      for c in json.loads((run / "candidates.json").read_text())}
        yield run.name, request, tifffile.imread(run / "image.tif"), tifffile.imread(run / "final_mask.tif") > 0, candidates
