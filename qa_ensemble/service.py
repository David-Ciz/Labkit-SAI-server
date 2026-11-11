from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
import logging
from time import perf_counter
from uuid import uuid4
import zipfile

import numpy as np
import tifffile

from .contracts import target_mask

logger = logging.getLogger(__name__)


class EnsembleService:
    def __init__(self, segment, pipelines, method_info, lock, recorder=None):
        self.segment = segment
        self.pipelines = pipelines
        self.method_info = method_info
        self.lock = lock
        self.recorder = recorder

    def run(self, image, request):
        pipeline = self.pipelines[request.pipeline_id]
        run_id = uuid4().hex
        started = perf_counter()
        source_results, candidates, statuses = {}, {}, []
        # This lock must also protect the original single-method endpoints.
        with self.lock:
            for method in request.methods:
                tick = perf_counter()
                status = {"method": method, "status": "failed"}
                try:
                    labels = np.asarray(self.segment(method, image.copy(), request))
                    if labels.shape != image.shape:
                        raise ValueError("Segmenter output shape differs from the ROI")
                    mask, match = target_mask(labels, request)
                    source_results[method] = labels
                    candidates[method] = mask
                    status.update(status="ok", **match)
                except Exception as exc:
                    # Preserve valid source outputs even when target matching failed.
                    if "labels" in locals() and labels.shape == image.shape and labels.dtype.kind in "uib":
                        source_results[method] = labels
                    status["error"] = str(exc)
                    logger.warning("Segmentation/target matching failed for %s: %s", method, exc)
                finally:
                    if "labels" in locals():
                        del labels
                status["elapsed_seconds"] = perf_counter() - tick
                status["provenance"] = self.method_info(method)
                statuses.append(status)

            result = {"scores": [], "diagnostics": {}, "stages": {}}
            if candidates:
                # The pipeline retains already completed QA outputs on later failure.
                result = pipeline.run(image, candidates, request)

        errors = any(s["status"] != "ok" for s in statuses)
        stage_errors = any(s.get("status") == "failed" for s in result.get("stages", {}).values())
        if not candidates:
            overall = "no_result"
        elif "ensemble_mask" not in result:
            overall = "partial"
        elif len(candidates) == 1:
            overall = "single_candidate" if not errors and not stage_errors else "partial"
        else:
            overall = "partial" if errors or stage_errors else "ok"
        recorded = False
        if self.recorder is not None and candidates:
            try:
                self.recorder.save(run_id, image, request, result.get("candidate_masks", candidates), result.get("scores", []))
                recorded = True
            except Exception as exc:  # recording must never fail the user's request
                logger.warning("Training record for %s was not saved: %s", run_id, exc)
        manifest = {
            "schema_version": 1, "run_id": run_id, "recorded": recorded, "image_id": request.image_id,
            "status": overall, "region": request.region.model_dump(),
            "roi_shape_yx": list(image.shape), "mask_coordinates": "context_roi",
            "coordinate_space": request.coordinate_space, "view_to_source_affine": request.view_to_source_affine,
            "methods": statuses, "pipeline": pipeline.provenance,
            "stages": result.get("stages", {}), "elapsed_seconds": perf_counter() - started,
            "warnings": result.get("warnings", []),
            "files": {},
        }
        return package_result(manifest, request.outputs, source_results, result.get("candidate_masks", candidates), result)


def package_result(manifest, outputs, sources, candidates, result):
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def write_tiff(key, filename, array):
            buf = BytesIO()
            array = np.asarray(array)
            if array.dtype == bool:
                array = array.astype("uint8")
            tifffile.imwrite(buf, array, photometric="minisblack")
            archive.writestr(filename, buf.getvalue())
            manifest["files"][key] = filename

        for key in ("ensemble_mask", "vote_map"):
            if key in outputs and key in result:
                write_tiff(key, f"{key}.tif", result[key])
        for key, mapping in (("source_segmentations", sources), ("candidate_masks", candidates)):
            if key in outputs:
                # Filenames are server-generated; method names never become paths.
                manifest["files"][key] = []
                for index, (method, array) in enumerate(mapping.items()):
                    filename = f"{key}/{index:03d}.tif"
                    write_tiff(f"_{key}_{index}", filename, array)
                    del manifest["files"][f"_{key}_{index}"]
                    manifest["files"][key].append({"method": method, "file": filename})
        columns = ["method"]
        if "qa_scores" in outputs:
            columns += ["qa_score"]
        if "selection" in outputs:
            columns += ["ranking_score", "retained", "selection_reason"]
        if "ensemble_weights" in outputs:
            columns += ["fusion_weight"]
        if len(columns) > 1:
            text = StringIO()
            writer = csv.DictWriter(text, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result.get("scores", []))
            archive.writestr("scores.csv", text.getvalue())
            manifest["files"]["scores"] = "scores.csv"
        if "diagnostics" in outputs:
            archive.writestr("diagnostics.json", json.dumps(result.get("diagnostics", {}), allow_nan=False))
            manifest["files"]["diagnostics"] = "diagnostics.json"
        archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False))
    return output.getvalue()
