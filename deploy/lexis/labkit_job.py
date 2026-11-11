#!/usr/bin/env python3
"""Labkit-SAI batch job for LEXIS AI Training (HEAppE GenericCommandTemplate).

Follows the ai-compute-templates convention: inputs under /mnt/input (read-only),
outputs under /mnt/output, one --mode switch, optional MLflow log in /mnt/output.

  --mode train    Train a pool-specific QA/ranker/fusion bundle.
                  /mnt/input/records/   records from an interactive server (LABKIT_RECORD_DIR), or
                  /mnt/input/images/ + /mnt/input/labels/   labelled TIFFs (candidates are generated here)
                  -> /mnt/output/bundles/<pipeline-id>/ (copy into the server's qa-bundles)
  --mode segment  Whole-image draft segmentation with QA filtering and fusion.
                  /mnt/input/images/  -> /mnt/output/segment/<name>_labels.tif, cells.csv, summary.json
                  Each cell proposed by --proposal-method is re-segmented by the pool from a box +
                  interior point, scored by QA and fused; low-QA cells are flagged, not hidden.

Both modes start the Labkit-SAI server inside the job on localhost with a random token.
Bundles come from /mnt/input/bundles if present, otherwise from the image's qa-bundles.

--finetune-cellpose (train, labelled images only) also fine-tunes Cellpose on the labels:
K image folds -> one model per fold -> each image's candidates come from the model that did
not see it -> QA/ranker/fusion learn from those out-of-fold candidates. A final model trained
on all images is written to /mnt/output/models/<pipeline-id>-ft.cellpose.torch.model and joins
the pool as method cellpose.<pipeline-id>-ft. Labels must label every cell in each image.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import zipfile

import click
import numpy as np
import requests
import tifffile
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
SIZE = 256


def log(*parts):
    print(*parts, flush=True)


@contextmanager
def local_server(bundles: Path | None, extra_models: Path | None, startup_timeout: float):
    """Run server-pytorch/server.py on a free localhost port for the duration of the job."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    token = secrets.token_hex(16)
    env = {**os.environ, "LABKIT_LOCAL_TEST_KEY": token, "PYTHONUNBUFFERED": "1"}
    if bundles is not None:
        env["LABKIT_QA_BUNDLES"] = str(bundles)
    if extra_models is not None:
        env["LABKIT_EXTRA_MODELS_DIR"] = os.pathsep.join(filter(None, [str(extra_models), env.get("LABKIT_EXTRA_MODELS_DIR", "")]))
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1",
                             "--port", str(port), "--workers", "1"], cwd=ROOT / "server-pytorch", env=env)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + startup_timeout
        while True:
            if proc.poll() is not None:
                raise SystemExit("Labkit-SAI server exited during startup")
            try:
                if requests.get(url + "/runtime", headers={"X-API-Token": token}, timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise SystemExit("Labkit-SAI server did not start in time")
            time.sleep(1)
        log(f"server ready at {url}")
        yield url, token
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def bundle_dir(input_root: Path) -> Path | None:
    candidate = input_root / "bundles"
    return candidate if candidate.is_dir() else None


def models_dir(input_root: Path) -> Path | None:
    """Extra segmenter files (e.g. a fine-tuned cellpose.<id>-ft from a train job)."""
    candidate = input_root / "models"
    return candidate if candidate.is_dir() else None


def mlflow_log(output: Path, experiment: str, params: dict, metrics: dict, artifacts: list[Path]):
    try:
        import mlflow
    except ImportError:
        log("mlflow not installed; skipping the MLflow log")
        return
    mlflow.set_tracking_uri(f"sqlite:///{output / 'mlflow.db'}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        for path in artifacts:
            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=path.name)
            elif path.exists():
                mlflow.log_artifact(str(path))


# ------------------------------------------------------------------ train
def finetune_folds(images: Path, labels: Path, work: Path, output: Path, pipeline_id, folds, epochs, seed, augment):
    """Per-fold Cellpose models for out-of-fold candidates, plus the final model on all images."""
    import hashlib
    names = sorted(p.name for p in images.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    k = min(folds, len(names))
    if k < 2:
        raise click.BadParameter("--finetune-cellpose needs at least two labelled images")
    order = sorted(names, key=lambda n: hashlib.sha256(f"{seed}|{n}".encode()).hexdigest())
    fold_of = {n: i % k for i, n in enumerate(order)}
    models = work / "models"
    models.mkdir(parents=True, exist_ok=True)
    (work / "folds.json").write_text(json.dumps(fold_of, indent=2))
    script = [sys.executable, str(ROOT / "scripts/finetune_cellpose.py"), "--images", str(images), "--labels", str(labels),
              "--epochs", str(epochs), "--seed", str(seed)] + (["--augment", augment] if augment else [])
    for fold in range(k):
        train_names = [n for n in names if fold_of[n] != fold]
        if not (models / f"ftfold{fold}.cellpose.torch.model").exists():
            log(f"fine-tuning Cellpose fold {fold}/{k} on {len(train_names)} images")
            subprocess.run(script + ["--name", f"ftfold{fold}", "--out-dir", str(models), "--only", *train_names], check=True)
    final = f"{pipeline_id}-ft"
    log(f"fine-tuning final Cellpose on all {len(names)} images")
    subprocess.run(script + ["--name", final, "--out-dir", str(output / "models")], check=True)
    return models, work / "folds.json", f"cellpose.{final}"


def run_train(input_root, output, methods, pipeline_id, base_bundle, silver_truth_src, raw_encoding,
              cells_per_image, max_epochs, seed, startup_timeout, finetune=False, ft_folds=3, ft_epochs=100, augment=None):
    records = input_root / "records"
    images, labels = input_root / "images", input_root / "labels"
    bundles = bundle_dir(input_root)
    base = Path(base_bundle) if base_bundle else None
    if base is None or not base.is_dir():
        search = [bundles] if bundles else []
        search.append(Path(os.getenv("LABKIT_QA_BUNDLES", ROOT / "qa-bundles")))
        base = next((root / base_bundle for root in search if base_bundle and (root / base_bundle).is_dir()), None)
        if base is None:
            raise click.BadParameter(f"Base bundle {base_bundle!r} not found in {search}")
    out = output / "bundles" / pipeline_id
    if out.exists():
        raise click.BadParameter(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/train_pool_bundle.py"), "--methods", ",".join(methods),
           "--base-bundle", str(base), "--pipeline-id", pipeline_id, "--out", str(out),
           "--work", str(output / "work" / pipeline_id), "--max-epochs", str(max_epochs), "--seed", str(seed),
           "--cells-per-image", str(cells_per_image), "--silver-truth-src", *silver_truth_src]
    if raw_encoding:
        cmd += ["--raw-encoding", raw_encoding]
    if finetune and not (images.is_dir() and labels.is_dir()):
        raise click.BadParameter("--finetune-cellpose needs fully labelled images/ + labels/ (records hold single cells)")
    if finetune:
        if "." in pipeline_id:
            raise click.BadParameter("--pipeline-id must not contain '.' with --finetune-cellpose")
        fold_models, fold_file, ft_method = finetune_folds(images, labels, output / "work" / pipeline_id, output,
                                                           pipeline_id, ft_folds, ft_epochs, seed, augment)
        methods = methods + [ft_method]
        cmd[cmd.index("--methods") + 1] = ",".join(methods)
        log(f"training QA/fusion on out-of-fold candidates; pool now {methods}")
        with local_server(bundles, fold_models, startup_timeout) as (url, token):
            subprocess.run(cmd + ["--images", str(images), "--labels", str(labels), "--server", url, "--token", token,
                                  "--oof-method", f"{ft_method}=cellpose.ftfold{{fold}}", "--fold-of", str(fold_file)], check=True)
    elif records.is_dir():
        log(f"training from interactive records in {records}")
        subprocess.run(cmd + ["--records", str(records)], check=True)
    elif images.is_dir() and labels.is_dir():
        log(f"training from labelled images in {images} / {labels}")
        with local_server(bundles, None, startup_timeout) as (url, token):
            subprocess.run(cmd + ["--images", str(images), "--labels", str(labels),
                                  "--server", url, "--token", token], check=True)
    else:
        raise click.BadParameter(f"Need {records}/ or both {images}/ and {labels}/")
    profile = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    exposure = profile.get("training_exposure", {})
    metrics = {"cells": exposure.get("cells", 0), "candidates": exposure.get("candidates", 0),
               "qa_inner_validation_mse": exposure.get("qa_inner_validation_mse", float("nan"))}
    for method, iou in exposure.get("candidate_mean_iou", {}).items():
        metrics[f"candidate_iou.{method}"] = iou
    mlflow_log(output, "labkit-train", {"pipeline_id": pipeline_id, "methods": ",".join(methods),
                                        "base_bundle": base.name, "source": exposure.get("source", ""),
                                        "finetune_cellpose": finetune}, metrics,
               [out] + ([output / "models"] if finetune else []))
    log(f"bundle written to {out}")


# ------------------------------------------------------------------ segment
def proposal_prompt(mask):
    """Simulated user prompt from a proposal: bbox + 10% margin + 3 px, point = interior maximum."""
    ys, xs = np.nonzero(mask)
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    my, mx = .10 * h + 3, .10 * w + 3
    box = [int(xs.min() - mx), int(ys.min() - my), int(xs.max() + 1 + mx), int(ys.max() + 1 + my)]
    dist = ndimage.distance_transform_edt(mask)
    py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return box, (int(px), int(py))


def roi_around(shape, px, py):
    h, w = shape
    x0 = int(np.clip(px - SIZE // 2, 0, max(0, w - SIZE)))
    y0 = int(np.clip(py - SIZE // 2, 0, max(0, h - SIZE)))
    return x0, y0, min(w, x0 + SIZE), min(h, y0 + SIZE)


def run_segment(input_root, output, methods, pipeline_id, proposal_method, flag_below, max_cells,
                min_area, startup_timeout, request_timeout):
    images = sorted(p for p in (input_root / "images").glob("*") if p.suffix.lower() in (".tif", ".tiff"))
    if not images:
        raise click.BadParameter(f"No TIFF images in {input_root / 'images'}")
    target = output / "segment"
    target.mkdir(parents=True, exist_ok=True)
    rows, totals = [], {"images": 0, "proposals": 0, "fused": 0, "flagged": 0, "failed": 0}
    with local_server(bundle_dir(input_root), models_dir(input_root), startup_timeout) as (url, token):
        headers = {"X-API-Token": token}
        caps = requests.get(url + "/segmentation_2D/qa_ensemble/capabilities", headers=headers, timeout=30).json()
        loaded = {p["pipeline_id"]: p for p in caps["pipelines"]}
        if pipeline_id not in loaded:
            raise click.BadParameter(f"Pipeline {pipeline_id!r} not loaded; available: {sorted(loaded)}")
        if not methods:  # default: the pool the bundle was trained on
            exposure = loaded[pipeline_id].get("training_exposure", {})
            methods = exposure.get("methods") or exposure.get("pool") if isinstance(exposure, dict) else None
            if not methods:
                raise click.BadParameter(f"Bundle {pipeline_id!r} does not record its pool; pass --methods")
            log(f"pool from bundle: {methods}")
        missing = set(methods) - {m["method_id"] for m in caps["methods"]}
        if missing:
            raise click.BadParameter(f"Segmenters not available on this server: {sorted(missing)} (fine-tuned models go in /mnt/input/models)")
        for path in images:
            raw = tifffile.imread(path)
            if raw.ndim != 2:
                log(f"skip {path.name}: not a 2D grayscale image")
                continue
            totals["images"] += 1
            buf = io.BytesIO()
            tifffile.imwrite(buf, raw.astype(np.float32))
            response = requests.post(f"{url}/segmentation_2D/on_posted_tiff/normalize_it_then_use/{proposal_method}",
                                     data=buf.getvalue(), headers={**headers, "Content-Type": "application/octet-stream"},
                                     timeout=request_timeout)
            response.raise_for_status()
            proposals = tifffile.imread(io.BytesIO(response.content))
            ids = [int(v) for v in np.unique(proposals) if v]
            if max_cells:
                ids = ids[:max_cells]
            labels = np.zeros(raw.shape, np.uint32)
            log(f"{path.name}: {len(ids)} proposals from {proposal_method}")
            for value in ids:
                totals["proposals"] += 1
                mask = proposals == value
                if mask.sum() < min_area:
                    continue
                box, (px, py) = proposal_prompt(mask)
                x0, y0, x1, y1 = roi_around(raw.shape, px, py)
                roi = raw[y0:y1, x0:x1].astype(np.float32)
                rbox = [max(0, box[0] - x0), max(0, box[1] - y0), min(x1 - x0, box[2] - x0), min(y1 - y0, box[3] - y0)]
                metadata = {"schema_version": 1, "image_id": path.stem[:256], "pipeline_id": pipeline_id,
                            "methods": methods, "return": ["ensemble_mask", "qa_scores", "selection", "ensemble_weights"],
                            "region": {"origin_xy": [x0, y0], "original_shape_yx": list(raw.shape),
                                       "target_box_xyxy": rbox, "target_point_xy": [px - x0, py - y0]}}
                rb = io.BytesIO()
                tifffile.imwrite(rb, roi)
                row = {"image": path.name, "proposal_label": value, "point_x": px, "point_y": py,
                       "status": "failed", "qa_top": "", "qa_weighted": "", "flagged": "", "output_label": ""}
                try:
                    r = requests.post(url + "/segmentation_2D/qa_ensemble", headers=headers, timeout=request_timeout,
                                      files={"image": ("roi.tif", rb.getvalue(), "image/tiff")},
                                      data={"metadata": json.dumps(metadata)})
                    r.raise_for_status()
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        manifest = json.loads(z.read("manifest.json"))
                        scores = list(csv.DictReader(io.StringIO(z.read("scores.csv").decode()))) if "scores.csv" in z.namelist() else []
                        fused = tifffile.imread(io.BytesIO(z.read("ensemble_mask.tif"))) > 0 if "ensemble_mask.tif" in z.namelist() else None
                    row["status"] = manifest["status"]
                    qa = [float(s["qa_score"]) for s in scores if s.get("qa_score") not in (None, "")]
                    weights = [(float(s["fusion_weight"]), float(s["qa_score"])) for s in scores
                               if s.get("fusion_weight") not in (None, "") and s.get("qa_score") not in (None, "")]
                    if qa:
                        row["qa_top"] = round(max(qa), 4)
                    if weights and sum(w for w, _ in weights) > 0:
                        row["qa_weighted"] = round(sum(w * q for w, q in weights) / sum(w for w, _ in weights), 4)
                    if fused is not None and fused.any():
                        label = int(labels.max()) + 1
                        region = labels[y0:y1, x0:x1]
                        region[fused & (region == 0)] = label  # first come keeps overlapping pixels
                        row["output_label"] = label
                        totals["fused"] += 1
                    gate = row["qa_weighted"] if row["qa_weighted"] != "" else row["qa_top"]
                    row["flagged"] = gate == "" or gate < flag_below
                    totals["flagged"] += bool(row["flagged"])
                except (requests.RequestException, KeyError, ValueError) as exc:
                    row["error"] = str(exc)[:200]
                    totals["failed"] += 1
                rows.append(row)
            tifffile.imwrite(target / f"{path.stem}_labels.tif", labels, compression="zlib")
    fields = ["image", "proposal_label", "point_x", "point_y", "status", "qa_top", "qa_weighted", "flagged", "output_label", "error"]
    with (target / "cells.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"pipeline_id": pipeline_id, "methods": methods, "proposal_method": proposal_method,
               "flag_below": flag_below, "flag_score": "fusion-weighted QA of retained candidates (not calibrated)", **totals}
    (target / "summary.json").write_text(json.dumps(summary, indent=2))
    mlflow_log(output, "labkit-segment", {k: summary[k] for k in ("pipeline_id", "proposal_method", "flag_below")},
               totals, [target / "cells.csv", target / "summary.json"])
    log(json.dumps(summary))


@click.command()
@click.option("--mode", type=click.Choice(["train", "segment"]), required=True)
@click.option("--input-root", type=click.Path(path_type=Path), default="/mnt/input", show_default=True)
@click.option("--output-root", type=click.Path(path_type=Path), default="/mnt/output", show_default=True)
@click.option("--methods", default=None,
              help="Comma-separated server method IDs: the segmenter pool. "
                   "segment: default = the bundle's recorded pool; train: default = Cellpose + SAM2 tiny/small/large.")
@click.option("--pipeline-id", default="labkit-pool-v1", show_default=True,
              help="segment: bundle to apply. train: ID of the new bundle.")
@click.option("--base-bundle", default="10-corrected-v2", show_default=True,
              help="train: bundle (directory name or path) providing QA initialisation and recipes.")
@click.option("--silver-truth-src", multiple=True, default=(), help="train: silver-truth src dir(s); default $SILVER_TRUTH_SRC (colon-separated).")
@click.option("--raw-encoding", type=click.Choice(["preserve", "minmax_uint8"]), default=None)
@click.option("--cells-per-image", default=8, show_default=True)
@click.option("--max-epochs", default=30, show_default=True)
@click.option("--seed", default=13, show_default=True)
@click.option("--proposal-method", default="cellpose.original", show_default=True, help="segment: whole-image proposer.")
@click.option("--flag-below", default=0.5, show_default=True, help="segment: flag cells whose QA is below this.")
@click.option("--max-cells", default=0, show_default=True, help="segment: limit cells per image (0 = all).")
@click.option("--min-area", default=20, show_default=True)
@click.option("--finetune-cellpose", is_flag=True, help="train: also fine-tune Cellpose (labelled images only).")
@click.option("--ft-folds", default=3, show_default=True, help="train: folds for out-of-fold Cellpose candidates.")
@click.option("--ft-epochs", default=100, show_default=True, help="train: Cellpose fine-tuning epochs (fixed recipe).")
@click.option("--augment", default=None, help="train: MODULE:FUNCTION extra offline augmentation for fine-tuning.")
@click.option("--startup-timeout", default=600.0, show_default=True)
@click.option("--request-timeout", default=600.0, show_default=True)
def main(mode, input_root, output_root, methods, pipeline_id, base_bundle, silver_truth_src, raw_encoding,
         cells_per_image, max_epochs, seed, proposal_method, flag_below, max_cells, min_area,
         finetune_cellpose, ft_folds, ft_epochs, augment, startup_timeout, request_timeout):
    """Labkit-SAI batch job (train | segment)."""
    methods = [m for m in (methods or "").split(",") if m]
    if mode == "train" and not methods:
        methods = ["cellpose.original", "sam2.original-tiny", "sam2.original-small", "sam2.original-large"]
    output_root.mkdir(parents=True, exist_ok=True)
    log(f"Labkit-SAI job: mode={mode} input={input_root} output={output_root}")
    if mode == "train":
        src = list(silver_truth_src) or [s for s in os.getenv("SILVER_TRUTH_SRC", "").split(":") if s]
        if not src:
            raise click.BadParameter("train needs --silver-truth-src or $SILVER_TRUTH_SRC (ranker/fusion trainers)")
        run_train(input_root, output_root, methods, pipeline_id, base_bundle, src, raw_encoding,
                  cells_per_image, max_epochs, seed, startup_timeout, finetune_cellpose, ft_folds, ft_epochs, augment)
    else:
        run_segment(input_root, output_root, methods, pipeline_id, proposal_method, flag_below, max_cells,
                    min_area, startup_timeout, request_timeout)


if __name__ == "__main__":
    main()
