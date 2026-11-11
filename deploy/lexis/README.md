# Labkit-SAI on LEXIS

There are two ways to run Labkit-SAI on IT4I/LEXIS. They share one container image (`labkit-sai.def`).

| Route | What it is | Entry point |
|---|---|---|
| **Interactive annotation** | The SAI server runs as an *engine* of the LEXIS/EXA4MIND AI Inference Service. The first request starts a GPU job. The service renews the job while it is used and lets it drain once idle. | `engine_start.sh` (`$1` port, `$2` pipeline ID, `$3` .sif) |
| **Batch jobs** | AI Training workflow block → HEAppE `GenericCommandTemplate` → `Labkit.sh` → `labkit_job.py` | `MODE=train` or `MODE=segment` |

## The training loop

1. **Annotate.** Start the server with `LABKIT_RECORD_DIR` set (opt-in, because it stores user images). Every QA/ensemble request then saves the ROI, the prompt and every target-matched candidate.
2. **Feed back.** After the user applies a mask, the client posts what was kept:
   ```
   POST /segmentation_2D/qa_ensemble/feedback   (multipart)
     run_id = <manifest.json run_id>
     action = applied | edited | rejected
     mask   = uint8 TIFF in the same context-ROI coordinates as the request (omit for rejected)
   ```
   `capabilities.recording` tells the client whether recording is on. The Labkit client (QA + ensemble dialog) sends `applied` with the previewed mask when the user clicks **Apply**, but only if the server recorded the run. Brush edits made after Apply are not captured yet.
3. **Train.** Run `MODE=train` with the record directory as the input dataset (`/mnt/input/records`). The stored candidates are reused, so no segmentation is re-run. Alternatively, give it labelled TIFFs (`/mnt/input/images` + `/mnt/input/labels`); the job then starts its own server to generate candidates.
4. **Serve.** Copy `/mnt/output/bundles/<pipeline-id>` into the engine's bundle directory (`LABKIT_BUNDLES`). The next engine job loads it.

**Fine-tuning Cellpose** (`--finetune-cellpose`, labelled images only):
1. The images are split into K folds, and one Cellpose model is fine-tuned per fold on the other folds.
2. Each image's Cellpose candidate comes from the model that did not see that image. QA, ranking and fusion are trained on these out-of-fold candidates.
3. A final model is trained on all images and written to `models/<pipeline-id>-ft.cellpose.torch.model`. It joins the pool as `cellpose.<pipeline-id>-ft`.

Labels must mark every cell in an image, because unmarked cells are learned as background. The recipe is fixed: cpsam base, lr 1e-5, weight decay 0.1, 100 epochs. `--augment MODULE:FUNCTION` adds offline augmentation, for example Vladimír's biomedical transforms.

To serve it, put the model file in the engine's `LABKIT_MODELS` folder, or in `/mnt/input/models/` for `segment`.

## Batch job I/O

| Mode | Input (`/mnt/input`) | Output (`/mnt/output`) |
|---|---|---|
| `train` | `records/` or `images/` + `labels/`; optional `bundles/` (base bundle) | `bundles/<pipeline-id>/`, `models/` (fine-tuned), `work/`, `mlflow.db` (only if mlflow is installed; the container env does not include it yet) |
| `segment` | `images/` (2D TIFF); optional `bundles/` and `models/`. The pool defaults to the bundle's recorded pool | `segment/<name>_labels.tif`, `segment/cells.csv`, `segment/summary.json` |

`segment` proposes cells with `--proposal-method`, then prompts the pool on each cell with a box and an interior point, and fuses the candidates. Cells are flagged when their fusion-weighted QA score is below `FLAG_BELOW`; flagged cells are kept, not removed. The threshold is not calibrated.

Run locally with the same layout:
```
python deploy/lexis/labkit_job.py --mode segment --input-root in/ --output-root out/ --pipeline-id labkit-pool-v1
```

## Engine notes

- `/metrics` answers without the token (`LABKIT_PUBLIC_METRICS=1`, liveness only). Every other route needs `X-API-Token`, which is read from `LABKIT_TOKEN_FILE`.
- **Open:** the inference service's proxy currently forwards only OpenAI-style routes. Labkit needs a pass-through route for `/segmentation_2D/...`, plus a decision on how the client's token and the LEXIS token combine. That is a service-side change; ask Adam.
- `LABKIT_REQUIRE_PIPELINE=$2` makes startup fail when the named bundle is missing, so the service never sees a healthy engine without its model.
- `LABKIT_WARMUP=1` loads every model of that bundle's pool before the server answers anything, including `/metrics`. Until then, the inference service keeps answering "model is being loaded", so no user request pays the roughly 60 s cold start. That matters because HEAppE's recommended API timeout is 60 s.

## Status (11 November 2026, updated)

- **Verified on the Mac (MPS)** with BF-C2DL-MuSC frames:
  - 14 simulated interactive requests were recorded, with feedback;
  - `train` from those records produced a bundle;
  - `segment` served that bundle (6 cells fused, 0 failures);
  - `train` from labelled images (in-job server) produced a bundle;
  - `train --finetune-cellpose` (10 silver-truth frames, 2 epochs, 3 folds) produced 3 fold models, a final model and a bundle over a 5-model pool;
  - `segment` with that bundle loaded the fine-tuned model and fused 3/3 cells.
- **Java client:** feedback compiles; 15 QA client tests pass; a live run against the FastAPI route recorded the run and accepted the feedback.
- **Not yet run:** the Apptainer build, `Labkit.sh` under HEAppE, `engine_start.sh` behind the inference service, and anything on Karolina/NVIDIA. These wait for an active IT4I project.
- **Known limits:**
  - the client sends `image_id = labkit-captured-view`, so record training falls back to one group per record (a weaker split);
  - fine-tuning on MPS needed `PYTORCH_ENABLE_MPS_FALLBACK=1`.
