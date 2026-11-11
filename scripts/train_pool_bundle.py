#!/usr/bin/env python3
"""Train a QA/ranker/fusion bundle for YOUR images and YOUR segmenter pool.

The pipeline only works on candidates like those it was trained on: same kind of
images, same segmenters, same prompting. This command

1. crops a 256x256 region around each annotated cell of your labelled images,
2. asks a running Labkit-SAI server to segment it with your chosen methods,
   prompted the way you will prompt in Labkit:
   --prompt cell (default): a box drawn around the cell (label bounding box + ~10% margin,
     jittered) with its centre as the foreground point, as the native client sends it;
   --prompt fixed: a central --box-size box + centre point (the MuSC labkit-pool-v1 setup),
3. fine-tunes the base bundle's QA on the resulting candidates (IoU against your
   labels), refits the pairwise ranker and the Deep Sets fusion with their
   original recipes, using out-of-fold QA scores,
4. exports a new frozen bundle for the server's qa-bundles directory.

Alternatively, --records trains from what users actually did in Labkit: a server
started with LABKIT_RECORD_DIR stores every candidate of each interactive request,
and the client's feedback call stores the mask the user applied. Those candidates
are reused as they are (no server needed), cropped 256x256 around the request's
anchor exactly as FrozenPipeline.run crops them; the applied mask is the reference.
Only records whose request included every --methods entry are used.

Example:
  python scripts/train_pool_bundle.py --images imgs/ --labels labels/ \
      --server http://127.0.0.1:8003 --token 1234 \
      --methods cellpose.original,sam2.original-tiny,sam2.original-small,sam2.original-large \
      --base-bundle qa-bundles/corrected-v2 --silver-truth-src ../silver-truth/src \
      --pipeline-id my-pool-v1 --out qa-bundles/my-pool-v1

Images are 2D grayscale TIFFs; labels are instance-label TIFFs with the same
file names (0 = background). Evaluated configurations: (a) BF-C2DL-MuSC, Cellpose +
three SAM2 sizes, --prompt fixed --box-size 96; (b) five CTC datasets, eight models
(Cellpose-SAM, Cellpose3, micro-SAM, SAM2.1-L box/point, StarDist, InstanSeg),
user-drawn box prompts, --raw-encoding minmax_uint8. See the evaluation reports.
Training needs the silver-truth research package (ranker/Deep Sets trainers).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import requests
import tifffile
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa_ensemble.frozen import Jaccard, make_features, rank_candidates, centered_crop  # noqa: E402
from qa_ensemble.contracts import RunRequest  # noqa: E402
from qa_ensemble.records import load_records  # noqa: E402

SIZE = 256
SUPPORT = ["ranker_cluster_full_boundary_contrast_rank", "ranker_cluster_rect_boundary_contrast_rank",
           "ranker_cluster_full_raw_contrast_norm_rank", "ranker_cluster_rect_raw_contrast_norm_rank"]


def log(*parts):
    print(*parts, flush=True)


# ---------------------------------------------------------------- data
def crop(array, cy, cx):
    y0, x0 = cy - SIZE // 2, cx - SIZE // 2
    out = np.zeros((SIZE, SIZE), array.dtype)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(array.shape[0], y0 + SIZE), min(array.shape[1], x0 + SIZE)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = array[sy0:sy1, sx0:sx1]
    return out


def cell_prompt(ref, key):
    """Box a user would draw around the cell (bbox + 10% +/-5% margin + 3 px) and its centre point."""
    ys, xs = np.nonzero(ref)
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    rng = np.random.default_rng(int(key, 16) % 2**32)
    j = lambda size: size * (.10 + rng.uniform(-.05, .05)) + 3
    box = [max(0, int(xs.min() - j(w))), max(0, int(ys.min() - j(h))),
           min(SIZE, int(xs.max() + 1 + j(w))), min(SIZE, int(ys.max() + 1 + j(h)))]
    return box, [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]


def collect_cells(images: Path, labels: Path, per_image: int, min_area: int, seed: int):
    rows, raws, refs = [], {}, {}
    key = lambda s: hashlib.sha256(f"{seed}|{s}".encode()).hexdigest()
    image_files = sorted(p for p in images.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    if not image_files:
        raise SystemExit("No TIFF images found")
    for order, path in enumerate(image_files):
        label_path = labels / path.name
        if not label_path.exists():
            raise SystemExit(f"Missing label image for {path.name}")
        raw, lab = tifffile.imread(path), tifffile.imread(label_path)
        if raw.ndim != 2 or raw.shape != lab.shape:
            raise SystemExit(f"{path.name}: expected matching 2D image and label shapes")
        ids = sorted((int(v) for v in np.unique(lab) if v), key=lambda v: key(f"{path.name}|{v}"))
        for value in ids[:per_image]:
            ys, xs = np.nonzero(lab == value)
            if len(ys) < min_area:
                continue
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            cid = f"{path.stem}__{value}"
            raws[cid] = crop(raw, cy, cx).astype(np.float32)
            refs[cid] = crop((lab == value).astype(np.uint8), cy, cx) > 0
            box, point = cell_prompt(refs[cid], key(f"{cid}|box"))
            # Images are the grouping unit for out-of-fold and inner splits.
            rows.append(dict(cell_id=cid, image=path.name, group=order, label=value, box=box, point=point))
    return pd.DataFrame(rows), raws, refs


def collect_records(root: Path, methods, min_area: int):
    """Cells, crops and candidates from interactive-session records (see qa_ensemble.records)."""
    rows, raws, refs, masks, status = [], {}, {}, {}, []
    groups, skipped = {}, {"methods": 0, "small": 0}
    for run_id, request, image, final, candidates in load_records(root):
        if not set(methods) <= set(request["methods"]):
            skipped["methods"] += 1
            continue
        if image.ndim != 2 or final.shape != image.shape:
            continue
        region = request["region"]
        x0, y0, x1, y1 = region["target_box_xyxy"]
        point = region.get("target_point_xy") or [(x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2]
        cx, cy = point
        ref = crop(final.astype(np.uint8), cy, cx) > 0
        if ref.sum() < min_area:
            skipped["small"] += 1
            continue
        ox, oy = cx - SIZE // 2, cy - SIZE // 2
        box = [int(np.clip(x0 - ox, 0, SIZE)), int(np.clip(y0 - oy, 0, SIZE)),
               int(np.clip(x1 - ox, 0, SIZE)), int(np.clip(y1 - oy, 0, SIZE))]
        raws[run_id] = crop(image, cy, cx).astype(np.float32)
        refs[run_id] = ref
        for method in methods:
            ok = method in candidates
            if ok:
                masks[(run_id, method)] = crop(candidates[method].astype(np.uint8), cy, cx) > 0
            status.append(dict(cell_id=run_id, method=method, status="ok" if ok else "failed", error=""))
        # The source image is the grouping unit for out-of-fold and inner splits.
        image_id = request.get("image_id", run_id)
        groups.setdefault(image_id, len(groups))
        rows.append(dict(cell_id=run_id, image=image_id, group=groups[image_id], label=1,
                         box=box, point=[cx - ox, cy - oy]))
    log(f"records: {len(rows)} usable; skipped {skipped['methods']} with a different method set, {skipped['small']} tiny masks")
    if not rows:
        raise SystemExit("No usable records: send feedback with the applied mask for runs using all --methods")
    cells = pd.DataFrame(rows)
    if len(groups) == 1 and len(cells) > 1:
        # Clients that do not name the source image (e.g. "labkit-captured-view") give no grouping;
        # fall back to one group per record. Cells of one image may then share folds' sides.
        log(f"records: all from image_id {next(iter(groups))!r}; grouping per record instead (weaker split)")
        cells["group"] = range(len(cells))
    return cells, raws, refs, masks, pd.DataFrame(status)


def generate_candidates(cells, raws, args, cache: Path):
    """Use the server's production route; candidate_masks are already target-matched.

    With --oof-method NAME=TEMPLATE, the candidate for NAME comes from the model
    TEMPLATE.format(fold=...) of the cell's image fold (a model not trained on that
    image), so QA/fusion never learn from a fine-tuned model's training images.
    """
    cache.mkdir(parents=True, exist_ok=True)
    masks, status = {}, []
    alias, fold_of = {}, {}
    if args.oof_method:
        name, _, template = args.oof_method.partition("=")
        fold_of = json.loads(Path(args.fold_of).read_text())
    for i, (cid, image, box, point) in enumerate(zip(cells.cell_id, cells.image, cells.box, cells.point)):
        requested = list(args.methods)
        if args.oof_method:
            if image not in fold_of:
                raise SystemExit(f"--fold-of has no fold for {image}")
            served = template.format(fold=fold_of[image])
            requested = [served if m == name else m for m in requested]
            alias = {served: name}
        target = cache / f"{cid}.zip"
        if not target.exists():
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, raws[cid])
            metadata = {"schema_version": 1, "image_id": cid[:256], "pipeline_id": args.request_pipeline,
                        "methods": requested, "return": ["candidate_masks", "qa_scores"],
                        "region": {"original_shape_yx": [SIZE, SIZE], "target_box_xyxy": box,
                                   "target_point_xy": point}}
            response = requests.post(args.server.rstrip("/") + "/segmentation_2D/qa_ensemble",
                                     headers={"X-API-Token": args.token}, timeout=args.timeout,
                                     files={"image": ("input.tif", buffer.getvalue(), "image/tiff")},
                                     data={"metadata": json.dumps(metadata)})
            response.raise_for_status()
            target.write_bytes(response.content)
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            ok = [m["method"] for m in manifest["methods"] if m["status"] == "ok"]
            for j, method in enumerate(ok):
                masks[(cid, alias.get(method, method))] = tifffile.imread(io.BytesIO(archive.read(f"candidate_masks/{j:03d}.tif"))) > 0
            status += [dict(cell_id=cid, method=alias.get(m["method"], m["method"]), status=m["status"],
                            error=m.get("error", "")) for m in manifest["methods"]]
        if (i + 1) % 25 == 0:
            log(f"candidates: {i + 1}/{len(cells)} cells")
    return masks, pd.DataFrame(status)


# ---------------------------------------------------------------- QA
class QATrainer:
    def __init__(self, base: Path, raws, masks, device, max_epochs, seed, raw_encoding=None):
        self.base, self.raws, self.masks = base, raws, masks
        self.device, self.max_epochs, self.seed = device, max_epochs, seed
        self.cfg = json.loads((base / "manifest.json").read_text())["qa"]
        if raw_encoding:
            self.cfg["raw_encoding"] = raw_encoding

    def encode(self, cid):
        raw = self.raws[cid]
        if self.cfg["raw_encoding"] == "minmax_uint8":  # same transform as FrozenPipeline.run
            span = float(raw.max() - raw.min())
            raw = np.zeros_like(raw) if span == 0 else ((raw - raw.min()) / span * 255).astype(np.uint8).astype(np.float32)
        return raw

    def tensors(self, df):
        x = np.stack([np.stack([self.encode(c), self.masks[(c, m)].astype(np.float32) * 255]) for c, m in zip(df.cell_id, df.method)])
        return torch.as_tensor(x), torch.as_tensor(df.true_iou.to_numpy(np.float32))

    def prep(self, x, augment):
        x = x.to(self.device).div(255.).sub(.5).div(.5)
        if augment:
            if torch.rand(1) < .5:
                x = x.flip(-1)
            if torch.rand(1) < .5:
                x = x.flip(-2)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), (-2, -1))
        return x

    def model(self):
        q = Jaccard(self.cfg["model_type"], self.cfg["dropout_rate"], self.cfg["output_activation"])
        q.load_state_dict(torch.load(self.base / "qa.pt", map_location="cpu", weights_only=True)["model_state_dict"])
        return q.to(self.device)

    def predict(self, q, df):
        x, _ = self.tensors(df)
        q.eval()
        with torch.inference_mode():
            return torch.cat([q(self.prep(x[i:i + 64], False)).flatten().cpu() for i in range(0, len(x), 64)]).numpy()

    def fit(self, df):
        """Original recipe: MSE, Adam 1e-4/1e-4, batch 16, flips/rot90, early stopping on held-out images."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        groups = np.array(sorted(df.group.unique()))
        if len(groups) >= 3:
            inner = set(rng.choice(groups, max(1, int(.15 * len(groups))), replace=False))
            train, val = df[~df.group.isin(inner)], df[df.group.isin(inner)]
        else:
            val = df.sample(frac=.15, random_state=self.seed)
            train = df.drop(val.index)
        xt, yt = self.tensors(train)
        xv, yv = self.tensors(val)
        q = self.model()
        optimizer = torch.optim.Adam(q.parameters(), lr=1e-4, weight_decay=1e-4)
        best, state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            q.train()
            for idx in torch.randperm(len(xt)).split(16):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(q(self.prep(xt[idx], True)).flatten(), yt[idx].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                optimizer.step()
            q.eval()
            with torch.inference_mode():
                pv = torch.cat([q(self.prep(xv[i:i + 64], False)).flatten().cpu() for i in range(0, len(xv), 64)])
            vl = float(((pv - yv) ** 2).mean())
            if vl < best - 1e-5:
                best, bad, state = vl, 0, {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}
            else:
                bad += 1
                if bad >= 5:
                    break
        q.load_state_dict(state)
        return q, best


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, help="labelled-image mode: grayscale TIFFs")
    p.add_argument("--labels", type=Path, help="labelled-image mode: instance-label TIFFs with the same names")
    p.add_argument("--records", type=Path, help="record mode: LABKIT_RECORD_DIR of a server (no --server needed)")
    p.add_argument("--server", help="labelled-image mode: running Labkit-SAI server URL")
    p.add_argument("--token", help="labelled-image mode: server API token")
    p.add_argument("--methods", required=True, help="comma-separated server method IDs (your pool)")
    p.add_argument("--base-bundle", type=Path, required=True, help="bundle providing QA init and recipes")
    p.add_argument("--silver-truth-src", type=Path, nargs="+", required=True,
                   help="silver-truth src directory (corrected candidate-context version first)")
    p.add_argument("--pipeline-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, help="cache directory (default: <out>.work)")
    p.add_argument("--request-pipeline", default="corrected-v2", help="any pipeline available on the server")
    p.add_argument("--cells-per-image", type=int, default=8)
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--prompt", choices=["cell", "fixed"], default="cell",
                   help="cell: box drawn around each cell, centre point (default); fixed: central --box-size box")
    p.add_argument("--box-size", type=int, default=96, help="--prompt fixed only (evaluated: 96)")
    p.add_argument("--raw-encoding", choices=["preserve", "minmax_uint8"],
                   help="QA intensity encoding (default: the base bundle's); use minmax_uint8 for 16-bit or float images")
    p.add_argument("--oof-method", help="NAME=TEMPLATE: serve NAME's candidates from TEMPLATE.format(fold=<image fold>)")
    p.add_argument("--fold-of", help="JSON file mapping image file name -> fold (with --oof-method)")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()
    args.methods = [m for m in args.methods.split(",") if m]
    if args.records is None and not all((args.images, args.labels, args.server, args.token)):
        p.error("Give --records, or all of --images --labels --server --token")
    if args.oof_method and (args.records is not None or not args.fold_of):
        p.error("--oof-method needs labelled-image mode and --fold-of")
    if args.out.exists():
        p.error("Output bundle exists; choose a new directory")
    work = args.work or args.out.with_name(args.out.name + ".work")
    work.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.silver_truth_src[0]))
    import silver_truth
    for extra in args.silver_truth_src[1:]:
        silver_truth.__path__.append(str(extra / "silver_truth"))
    from silver_truth.evaluation import candidate_set_weight as cw
    from silver_truth.evaluation.candidate_ranker import add_cluster_ranker_features
    from silver_truth.evaluation.cluster_pairwise_ranker import ClusterPairwiseRankerConfig, train_cluster_pairwise_ranker

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.records is not None:
        cells, raws, refs, masks, status = collect_records(args.records, args.methods, args.min_area)
        args.prompt = "recorded"
    else:
        cells, raws, refs = collect_cells(args.images, args.labels, args.cells_per_image, args.min_area, args.seed)
        if args.prompt == "fixed":
            half = args.box_size // 2
            cells["box"] = [[SIZE // 2 - half, SIZE // 2 - half, SIZE // 2 + half, SIZE // 2 + half]] * len(cells)
            cells["point"] = [[SIZE // 2, SIZE // 2]] * len(cells)
        log(f"{len(cells)} labelled cells from {cells.image.nunique()} images")
        masks, status = generate_candidates(cells, raws, args, work / "candidates")
    status.to_csv(work / "candidate-status.csv", index=False)
    rows = [dict(cell_id=c, method=m, true_iou=float((mk & refs[c]).sum() / max(1, (mk | refs[c]).sum())))
            for (c, m), mk in masks.items()]
    cand = pd.DataFrame(rows).merge(cells, on="cell_id")
    cand["split"] = "train"
    log("candidate mean IoU by method:", cand.groupby("method").true_iou.mean().round(3).to_dict())
    if cand.cell_id.nunique() < 10:
        p.error("Fewer than 10 cells have candidates; annotate more cells")

    # QA: out-of-fold scores (image blocks) for downstream training, then a final model.
    qa = QATrainer(args.base_bundle, raws, masks, device, args.max_epochs, args.seed, args.raw_encoding)
    groups = np.array(sorted(cand.group.unique()))
    cand["qa_oof"] = np.nan
    for block in [b for b in np.array_split(groups, min(5, len(groups))) if len(b)]:
        hold = cand.group.isin(block)
        if hold.all():
            continue
        model, _ = qa.fit(cand[~hold])
        cand.loc[hold, "qa_oof"] = qa.predict(model, cand[hold])
    if cand.qa_oof.isna().any():  # single-image input: no out-of-fold estimate possible
        p.error("Provide labelled cells from at least two images")
    final, inner_mse = qa.fit(cand)
    log(f"QA fine-tuned; inner-validation MSE {inner_mse:.4f}")

    # Features exactly as the server computes them.
    manifest = json.loads((args.base_bundle / "manifest.json").read_text())
    if args.raw_encoding:
        manifest["qa"] = {**manifest["qa"], "raw_encoding": args.raw_encoding}
    prompts = cells.set_index("cell_id")
    features = []
    for cid, g in cand.groupby("cell_id", sort=False):
        request = RunRequest(region={"original_shape_yx": (SIZE, SIZE), "target_point_xy": tuple(prompts.point[cid]),
                                     "target_box_xyxy": tuple(prompts.box[cid])},
                             methods=sorted(g.method), pipeline_id=args.pipeline_id)
        raw = qa.encode(cid)
        crop_raw, box = centered_crop(raw, request.anchor, SIZE)
        candidate_masks = {m: masks[(cid, m)] for m in g.method}
        f = make_features(raw, crop_raw, candidate_masks, candidate_masks, dict(zip(g.method, g.qa_oof)), request, box, manifest)
        f = f.merge(g[["method", "true_iou", "split", "label"]].rename(columns={"method": "candidate_source"}), on="candidate_source")
        f["image_id"] = f["candidate_set_id"] = f["cell_id"] = cid
        features.append(f)
    features = pd.concat(features, ignore_index=True)
    features["fold"], features["campaign_number"] = "pool", "user"

    base_ranker = joblib.load(args.base_bundle / "ranker.joblib")
    augmented = add_cluster_ranker_features(features)
    ranker_cfg = ClusterPairwiseRankerConfig(**{**base_ranker["config"], "train_splits": ("train",)})
    ranker = train_cluster_pairwise_ranker(augmented, config=ranker_cfg)
    joblib.dump(ranker, work / "ranker.joblib")

    selected = []
    for _, g in augmented.groupby("cell_id", sort=False):
        ranked = rank_candidates(g, ranker)
        ranked["local_image_support_score"] = ranked.reindex(columns=SUPPORT).mean(axis=1).fillna(.5).clip(0, 1)
        selected.append(ranked.sort_values("pairwise_cluster_score", ascending=False, kind="stable")
                        .head(manifest["ensemble"]["max_candidates"]))
    selected = pd.concat(selected, ignore_index=True)
    base_ensemble = torch.load(args.base_bundle / "ensemble.pt", map_location="cpu", weights_only=False)
    ens_cfg = replace(cw.CandidateSetWeightConfig(**base_ensemble["config"]), random_state=args.seed,
                      deep_sets_device="cpu", train_splits=("train",))
    torch.manual_seed(args.seed)
    if (work / "ensemble.joblib").exists():
        (work / "ensemble.joblib").unlink()
    ensemble = cw.train_candidate_set_weight_predictor(selected, config=ens_cfg, output_model=work / "ensemble.joblib")
    log(f"ranker and fusion refitted on {len(selected)} selected candidates / {ensemble['train_clusters']} cells")

    torch.save({"model_state_dict": {k: v.cpu() for k, v in final.state_dict().items()}}, work / "qa.pt")
    profile = {key: manifest[key] for key in ("qa", "ensemble", "dataset_identity")}
    profile["pipeline_id"] = args.pipeline_id
    profile["training_exposure"] = {
        "status": "pool-specific training with scripts/train_pool_bundle.py",
        "source": "interactive records" if args.records is not None else "labelled images",
        "base_bundle": manifest["pipeline_id"], "methods": args.methods, "prompt": args.prompt,
        "box_size": args.box_size if args.prompt == "fixed" else None,
        "out_of_fold_method": args.oof_method,
        "images": int(cells.image.nunique()), "cells": int(cand.cell_id.nunique()), "candidates": int(len(cand)),
        "candidate_mean_iou": cand.groupby("method").true_iou.mean().round(4).to_dict(),
        "qa_inner_validation_mse": inner_mse, "seed": args.seed}
    (work / "profile.json").write_text(json.dumps(profile, indent=2))
    subprocess.run([sys.executable, str(ROOT / "scripts/export_qa_bundle.py"), "--qa", str(work / "qa.pt"),
                    "--ranker", str(work / "ranker.joblib"), "--ensemble", str(work / "ensemble.joblib"),
                    "--profile", str(work / "profile.json"), "--output", str(args.out)], check=True)
    log(f"Done. Copy {args.out} into the server's qa-bundles directory and restart the server.")


if __name__ == "__main__":
    main()
