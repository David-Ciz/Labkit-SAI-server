#!/usr/bin/env python3
"""Fine-tune Cellpose(-SAM) on fully labelled images and write a server model file.

The output is named <name>.cellpose.torch.model, which the server discovers as the
method "cellpose.<name>" (put it in LABKIT_MODELS_DIR or LABKIT_EXTRA_MODELS_DIR).

Labels must be complete instance masks: every cell in a training image labelled.
Unlabelled cells are learned as background, so sparse annotations (e.g. CTC GT
frames with only some cells marked) degrade the model.

Recipe (fixed, not tuned per dataset): cpsam base, AdamW lr 1e-5, weight decay 0.1,
batch 1, 256 px tiles, Cellpose's own rotation/flip/scale augmentation.
--augment MODULE:FUNCTION adds offline augmentation: FUNCTION(image, labels) returns
a list of extra (image, labels) pairs (e.g. biomedical intensity/noise transforms).
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import tifffile


def load_pairs(images: Path, labels: Path, names=None):
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    if names is not None:
        names = set(names)
        files = [p for p in files if p.name in names]
    pairs = []
    for path in files:
        lab = labels / path.name
        if not lab.exists():
            raise SystemExit(f"Missing label image for {path.name}")
        image, label = tifffile.imread(path), tifffile.imread(lab)
        if image.ndim != 2 or image.shape != label.shape:
            raise SystemExit(f"{path.name}: expected matching 2D image and label shapes")
        pairs.append((image.astype(np.float32), label.astype(np.int32)))
    return pairs


def load_augment(spec):
    if not spec:
        return None
    module, _, function = spec.partition(":")
    return getattr(importlib.import_module(module), function or "augment")


def finetune(pairs, out_file: Path, epochs=100, learning_rate=1e-5, weight_decay=0.1, seed=13,
             augment=None, base="cpsam", device=None):
    import torch
    from cellpose import models, train
    if not pairs:
        raise SystemExit("No training images")
    if augment is not None:
        pairs = pairs + [extra for image, label in pairs for extra in augment(image, label)]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = models.CellposeModel(device=device, pretrained_model=base)
    with tempfile.TemporaryDirectory() as tmp:
        saved, losses, _ = train.train_seg(
            model.net, train_data=[p[0] for p in pairs], train_labels=[p[1] for p in pairs],
            batch_size=1, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=epochs,
            normalize=True, save_path=tmp, save_every=10**9, min_train_masks=1, model_name="finetuned")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(saved, out_file)
    return {"images": len(pairs), "epochs": epochs, "learning_rate": learning_rate, "weight_decay": weight_decay,
            "base": base, "seed": seed, "final_train_loss": float(losses[-1]) if len(losses) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--only", nargs="*", help="train on these image file names only")
    p.add_argument("--name", required=True, help="model name; the server method becomes cellpose.<name>")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--augment", help="MODULE:FUNCTION for extra offline augmentation")
    args = p.parse_args()
    if "." in args.name:
        p.error("--name must not contain '.'")
    out = args.out_dir / f"{args.name}.cellpose.torch.model"
    info = finetune(load_pairs(args.images, args.labels, args.only), out, args.epochs, args.learning_rate,
                    args.weight_decay, args.seed, load_augment(args.augment))
    info.update(method=f"cellpose.{args.name}", file=str(out), trained_on=args.only or "all")
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    sys.exit(main())
