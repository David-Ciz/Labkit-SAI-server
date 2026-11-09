# Remote QA and ensemble

The combined PyTorch endpoint runs the selected segmentation methods, extracts one target-cell mask from each output, applies a frozen QA/ranker/Deep Sets bundle and returns the requested results. TensorFlow/StarDist remains a separate optional service. No training occurs in the server or client.

**Current native deployment — 4 November 2026:** `corrected-v2` is installed and active alongside the preserved legacy pipeline. The exporter and adapter support candidate-grouping version 2, recompute context-isolated relative features and infer each candidate set separately. Research/server parity is exact on 10,598 rows; 29 server tests and a real four-model MPS request pass. David's separate GUI preview/apply check also passes. Accuracy remains limited on the checked example (both bundles IoU 0.2935, Cellpose 0.5158). See `/Users/davidciz/second-brain/outputs/labkit-corrected-bundle-2026-11-04/README.md`. The dated records below describe earlier stages; Linux GPU/HPC verification remains open.

## Current implementation and remaining inputs

- Implemented: authenticated combined endpoint, capabilities, one-cell target matching, frozen inference adapter, optional output bundle, partial results, shared model locks, container configuration and a Python client.
- Native client changes are in the `qa-ensemble` checkout based on `xulman/labkit-ui`, branch `SAI-on-bigdataviewer-prompts`, commit `636d5c047612bd6548304f92b3c3f8b7c01a778e`. The SAI controller has a **QA + ensemble...** dialog with method/output selection, prompt capture, remote execution, Fiji previews and explicit application to labels. The complete modified client compiles and packages successfully using Java 11 or newer. The existing `sc.fiji:bigdataviewer-prompts:0.0.2` dependency was recovered from the [xulman Fiji update site](https://sites.imagej.net/xulman/) and installed into the local Maven cache with its embedded POM. It supplies the pre-existing BigDataViewer prompt/view interaction layer. The handoff evidence includes the test plugin, dependency provenance and build logs. Client tests were skipped; live Fiji interaction remains unverified. The installed Fiji application was not changed.
- Recovered: MuSC fold-1 ResNet18 QA checkpoint from the existing Karolina experiments; existing Deep Sets payloads remain in the silver-truth checkout.
- Replacement ranker: David authorized training one when the original checkpoint could not be located. A fixed MuSC fold-1 refit uses 691 training candidates / 174 cell groups / 1,704 pairs and reproduces all 2,042 retained ranking scores to within 9e-13. Validation/test rows were excluded from fitting; QA and ensemble weights remain unchanged. The experimental local bundle is `qa-bundles/ctc-musc-fold1-ranker20261016-experimental-v1`. Its real QA/ranker/ensemble inference and output packaging pass a synthetic-mask smoke check, including unknown method IDs and missing context features. The evidence directory contains training/input provenance and validation reports.
- A fixed ten-cell CTC development pilot is now complete (below). A target deployment host and native prompt/apply verification remain open. The profile preserves the old QA checkpoint's missing activation metadata; April 14 source history supports the deployed linear setting. No new-domain accuracy, GPU inference or remote demo is claimed.
- The server discovers complete valid bundles at startup. The installed experimental bundle is available on the next configured startup; the running service was not restarted. With no complete valid bundle, it reports no available pipelines. A request for an unavailable pipeline returns HTTP 503. It does not train a replacement ranker or silently substitute another fusion method.

Component verification: the recovered QA checkpoint produces exactly the same output in the original and extracted inference definitions on a fixed synthetic input. The ensemble reproduces all 300 saved candidate scores in 100 cell groups with maximum absolute error 0.0. The replacement ranker also matches the original scoring code exactly on 31 candidates in eight validation cell groups. These checks do not establish full historical-pipeline parity or Labkit transfer accuracy.

## Fixed CTC development pilot — 19 October 2026

Ten preselected held-out MuSC crops completed SAM2 tiny + Cellpose → frozen QA/ranking/ensemble. Nine requests fused two candidates; one returned a single SAM2 candidate after Cellpose found no target. Mean IoU including the missing Cellpose mask as zero: Cellpose 0.6177, SAM2 0.3585, QA selection 0.4426, ensemble 0.5504, equal vote 0.4504. QA ordered 4/9 comparable pairs correctly. Ranker selection reached 0.6500, matching the simple Cellpose-first/SAM2-fallback baseline. This does not establish a learned-ranking advantage. No fitting or tuning occurred; warm CPU requests averaged 83.1 s. The frozen fusion improves four cells beyond both candidates but is worse than Cellpose on average, with severe QA/fusion failures. Do not treat this as validated automatic mask acceptance.

The original QA and adapter agree exactly on ten hash-restored crops. April 14 repository source returns a linear head directly, supporting the deployed setting. Archived predictions differ by at most 0.000226 (above the initial strict 0.0001 tolerance); the cause is unresolved. A separate historical ranker/ensemble input-table provenance gap remains recorded. Report, all-cell overlays, per-cell scores, fixed protocol and evidence hashes: `/Users/davidciz/second-brain/outputs/labkit-qa-ensemble-2026-10-19/qa-sanity/README.md`. Native held-L prompt capture and applying after viewer movement remain unverified; home GPU and remote deployment remain open.

## Local real-model trial — 19 October 2026

A separate development container at `http://127.0.0.1:8002` mounts the updated source/bundle into the existing locked image. The existing port-8000 service remains unchanged. Real SAM2 tiny and small → QA/ranker/ensemble requests pass on the CC0 scikit-image quantitative-phase cell ROI. Cold time is 112.3 s including model loading; the built Java client's Float32 TIFF multipart upload and ZIP/TIFF decode pass in 20.2 s (20.0 s server time). The separate ImageJ/Labkit session opens and displays the server, experimental pipeline, method choices and output controls. Held-L prompt capture and preview-to-label placement remain unverified.

The first request's QA raw outputs are -0.0102/-0.0084, preserved without clipping. No calibrated IoU or new-domain accuracy is claimed. The subsequent ten-crop check reproduces original preprocessing and inference exactly, and historical source supports the linear head; archived predictions differ by up to 0.000226. No QA/ensemble fitting or threshold changes were made. This local CPU check does not complete the home GPU-machine trial or remote deployment. Results and instructions: `/Users/davidciz/second-brain/outputs/labkit-qa-ensemble-2026-10-19/local-trial/README.md`.

## Install a frozen bundle

Use already-trained artifacts from one compatible original run. Run the exporter in a Python environment containing PyTorch and joblib:

```sh
python scripts/export_qa_bundle.py \
  --qa /path/to/saved_qa.pt \
  --ranker /path/to/saved_pairwise_ranker.joblib \
  --ensemble /path/to/saved_deep_sets.joblib \
  --profile examples/frozen-profile.json \
  --output qa-bundles/ctc-musc-fold1-frozen-v1
```

Edit the example profile from the matching experiment before exporting. Record training/held-out exposure. Its activation and intensity/crop settings are explicit: the recovered older QA checkpoint lacks an activation entry, and `linear` reproduces the current original loader's compatibility default. That is not evidence that every historical checkpoint used a linear training head. Preserve the actual crop encoding and tensor transform; do not assume arbitrary 0–1 images are equivalent to training intensities. The optional `minmax_uint8` raw encoding must be selected deliberately for an appropriate profile; the default example preserves raw values.

Each bundle directory contains `manifest.json`, `qa.pt`, `ranker.joblib` and `ensemble.pt`. Hashes are checked at load time. Artifacts are trusted administrator-installed files, never API uploads; sklearn joblib files are executable serialization. The bundle directory is excluded from git and Docker build context and mounted read-only at runtime.

Set `LABKIT_QA_BUNDLES` to the bundle root and `LABKIT_QA_DEVICE` to `cpu` (default) or `cuda`. Pipelines load at application startup; restart after installing/updating one. The supplied Compose configuration mounts `./qa-bundles` automatically. Existing single-method endpoints continue working without any bundle.

For optional StarDist orchestration, create the TensorFlow token as described in the main README and use:

```sh
docker compose -f compose.yaml -f compose.qa-tensorflow.yaml --profile tensorflow up -d --build --wait
```

On a configured NVIDIA host, include `-f compose.gpu.yaml` as well. This document does not identify or deploy to a particular remote host. Model inference and GPU compatibility must be checked there.

## Train a bundle for your images and segmenter pool

QA, ranking and fusion only work on candidates like the ones they were trained on. A bundle trained on other
segmenters can rank your candidates **in reverse**. In the BF-C2DL-MuSC evaluation, the CTC-trained bundle
ordered Cellpose/SAM2 pairs correctly only 36.6% of the time; after pool-specific training, 82.4%. So train a
bundle whenever you change the image type, the method list or the prompting style:

```sh
python scripts/train_pool_bundle.py --images imgs/ --labels labels/ \
    --server http://127.0.0.1:8003 --token <key> \
    --methods cellpose.original,cellpose3.cyto3,microsam.vit_b_lm,sam2.original-large,sam2.original-large-point,stardist.versatile-fluo \
    --raw-encoding minmax_uint8 \
    --base-bundle qa-bundles/corrected-v2 \
    --silver-truth-src <silver-truth candidate-context-v2>/src <silver-truth>/src \
    --pipeline-id my-pool-v1 --out qa-bundles/my-pool-v1
```

- **Inputs:** 2D grayscale TIFFs plus instance-label TIFFs with the same file names. Cells from at least two
  images are required; how many you need is reported in the evaluation's label-budget addendum.
- **Candidates** come from the running server's own `/segmentation_2D/qa_ensemble` route. `--prompt cell`
  (default) sends a box drawn around each annotated cell (bounding box + ~10% jittered margin) with its centre
  as the foreground point, as the native client does when you draw a box. `--prompt fixed --box-size 96`
  reproduces the central-box setup of `labkit-pool-v1`. Use the prompting you will use interactively.
- **Intensities:** `--raw-encoding minmax_uint8` rescales each crop for QA; use it for 16-bit or float images.
- **Training:** QA is fine-tuned from the base bundle (MSE, Adam 1e-4, flips/rot90, early stopping on held-out
  images). Out-of-fold QA scores feed the pairwise ranker and Deep Sets refits, which use the base bundle's
  recipes. Thresholds, temperature and top-k are inherited.
- Responses are cached in `<out>.work/` so an interrupted run resumes. Copy the bundle into the server's
  bundle directory and restart; `corrected-v2` and other bundles stay selectable for rollback.
- Fusion cannot exceed what the pool offers. With one strong model and several weak, correlated ones, expect
  parity with the best model, plus calibrated QA scores and safe fallback, rather than large gains. With a
  diverse pool (the six-model pool above, five CTC datasets, user-drawn boxes) the fused result beat the single
  model chosen on the training cells by +0.081 IoU and QA flagged bad masks with AUROC 0.91
  (`labkit-evaluation/model-pilot-2026-11-07/README.md`).

### Models in separate environments (side servers)

micro-SAM, Cellpose 3 and StarDist cannot share the PyTorch server's environment (Cellpose 3 conflicts with
Cellpose 4; StarDist needs TensorFlow). `server-extra/side_server.py` serves them from their own environments;
the main server forwards to them and advertises their methods:

| Method ID | Side group | Prompt |
|---|---|---|
| `microsam.vit_b_lm` | `microsam` | box + point |
| `cellpose3.cyto3`, `cellpose3.nuclei` | `cellpose3` | crop |
| `stardist.versatile-fluo` | `stardist` (TensorFlow CPU) | crop |
| `sam2.original-large-point` (in the main server) | — | point only |

Start each with `<env python> server-extra/side_server.py <group> --port <p> --token-file <file>` and set
`LABKIT_SIDE_BACKENDS="microsam=http://127.0.0.1:8011,cellpose3=http://127.0.0.1:8012,stardist=http://127.0.0.1:8013"`
and `LABKIT_SIDE_TOKEN_FILE` for the main server. On the Mac, `runtime/side-backends.json`
(`{"<group>": {"python": "<env>/bin/python", "port": 8011}}`) makes `scripts/macos_server.py start/stop`
manage them with a fresh token. `sam2.original-large-point` is advertised only when the empty marker
`server-MODELS/original-large-point.sam2.torch.model` exists (`touch` it; `*.model` is git-ignored). An unreachable side server simply advertises no methods. Served masks equal the
study wrappers' output exactly, limited to the QA window like every candidate (checked on 50 cells × 8 models).

## Request and response

`GET /segmentation_2D/qa_ensemble/capabilities` lists method IDs, prompt support, available frozen pipelines, output names and configured limits. The PyTorch service advertises StarDist models from the mounted model directory only when its internal TensorFlow URL is configured. Actual backend failures are returned per method.

`POST /segmentation_2D/qa_ensemble` accepts multipart fields:

- `image`: one 2D grayscale TIFF page, retaining its original intensity values.
- `metadata`: JSON matching `examples/qa-request.json`.

Both endpoints require the existing `X-API-Token` header. The CLI reads it from a file:

```sh
python scripts/qa_ensemble_client.py \
  --server http://127.0.0.1:8000 \
  --token-file server-pytorch/secret.txt \
  --image /path/to/context_roi.tif \
  --metadata examples/qa-request.json \
  --output /path/to/result.zip
```

Use real image dimensions and target coordinates in the metadata. The target box is half-open `[x0,y0,x1,y1]`, with x=column and y=row, in the uploaded context ROI. `origin_xy` locates that ROI within `original_shape_yx`. An optional target point defaults to the box center. Source-image requests use `coordinate_space: source_image`. The native SAI client submits its captured view with `coordinate_space: captured_view` and preserves the view-to-label-source affine transform; it applies the returned mask through the saved prompt's existing native coordinate mapping.

SAM2 receives the target box. Other current wrappers segment their supplied context ROI. The target matcher first uses the label containing the anchor, then permits a single unambiguous label intersecting the target box. Several possible labels without an anchor match are reported as ambiguous; they are not fused together as one cell.

The `return` list selects:

| Value | Returned data |
| --- | --- |
| `ensemble_mask` | Final binary mask, in context-ROI coordinates |
| `source_segmentations` | Original method label images |
| `candidate_masks` | The specific masks actually assessed/fused, limited to the frozen QA crop |
| `qa_scores` | Predicted mask-quality values |
| `selection` | Pairwise ranking, retained/excluded state and reason |
| `ensemble_weights` | Normalized fusion contribution |
| `vote_map` | Continuous weighted-vote strength; not a calibrated probability |
| `diagnostics` | Missing-feature handling, unknown sources and crop details |

The ZIP always includes `manifest.json` with coordinates, pipeline/artifact provenance, per-method and per-stage status, warnings and file references. Score choices share `scores.csv`; only requested columns are included. Images/maps are TIFFs. Output choices filter returned data, while intermediate calculations needed by the ensemble still run. QA scores, ranking scores and fusion weights are distinct. Numeric quality estimates are not measured accuracy.

The manifest distinguishes `ok`, `partial`, `single_candidate` and `no_result`. Failure of one method preserves valid outputs from the others. Failure of ranking/ensemble preserves completed QA and requested source results. One valid candidate is identified as a single-candidate result. Empty or failed stages never receive a fabricated ensemble mask.

## Frozen input adaptation

The adapter preserves the original feature math and stores source hashes in `qa_ensemble/vendor_provenance.json`. New Labkit methods retain their real IDs. Existing Deep Sets handling encodes unseen categorical values as all-zero vectors and imputes missing numeric features from its saved training means. The saved sklearn ranker applies its own stored preprocessing; a compatible artifact must support the intended missing/unknown inputs.

Image/mask geometry, candidate-relative ranks and supported image contrast are computed directly. Whole-image population ranks, unknown source priors and absent temporal information remain missing. Full-context measurements are computed only where the uploaded context is sufficient. This behavior is reported in diagnostics rather than replacing missing measurements with invented source statistics or evaluated reference labels.

The QA crop is centered at the target anchor with the profile's original fixed size and zero padding. If a candidate extends beyond it, the returned ensemble is limited to that crop, and a manifest warning requests inspection before applying. The adapter never resizes the resulting mask to an unrelated region. A new profile must be selected deliberately if the task needs a different crop/preprocessing policy.

## Execution limits and checks

Selected models run sequentially under the same lock as existing single-method inference and model-list updates. This protects the process-local mutable model cache. QA and ensemble remain resident. TensorFlow uses its own token internally.

Configurable limits: `LABKIT_QA_MAX_PIXELS` (4,194,304), `LABKIT_QA_MAX_BYTES` (67,108,864), `LABKIT_QA_MAX_METHODS` (8), `LABKIT_TENSORFLOW_TIMEOUT` (180 seconds). The Python and native clients use a 600-second response timeout. A client timeout does not cancel already-running local GPU inference; measured target-host runtimes determine whether asynchronous jobs are needed later.

Run tests in a development environment with the inference dependencies plus pytest and httpx:

```sh
python -m pytest tests -q
```

The suite covers request validation/authentication, selected outputs, ROI mapping, target ambiguity, one/zero candidates, backend/stage failures, missing/unseen inputs and concurrent model selection. Synthetic test doubles verify plumbing; they are not learned-method results.

`scripts/verify_frozen_parity.py` compares the recovered QA implementation with its original source and Deep Sets outputs with a saved candidate table. It takes explicit artifact/source paths and writes a JSON report; it performs no fitting or model downloads. Before a scientific pilot, also verify the recovered ranker and the complete matching inference bundle, then run the fixed image batch with independent reference masks only in the external evaluator.

## Isolated target-validation branch

`codex/labkit-target-validation` adds two rules without changing the original running service or frozen artifacts. An explicitly supplied `target_point_xy` is a foreground assertion: a prediction missing it is rejected, and a single foreground label cannot override that assertion. Without an explicit point, the box-center fallback remains, but a fragmented binary mask is marked ambiguous rather than treated as one unambiguous instance. Requested source segmentations remain available even if target matching fails.

The combined SAM2 path forwards an explicit point as a positive prompt in addition to the box. A derived box center is not automatically promoted to a positive point. Legacy box-only segmentation calls retain their existing behavior. Point inclusion is a necessary correspondence check when the user asserts foreground; it is not a guarantee of accurate boundaries or mask quality. The diagnostic point-only additions already showed that poor masks can contain the point. The cached-mask rejection result (0.6683) is not an accuracy claim for this changed prompt path.

The research grouping correction lives on `codex/ensemble-candidate-context-v2` in the silver-truth repository. Corrected version-2 weights were trained and integrated through the compatible export/adapter on 4 November (see the current deployment note above); the historical bundle is not relabelled or overwritten. The QA checkpoint's poor stress-control behavior is not fixed by these engineering changes.

## Multi-model execution and checkpoint preparation

Model switching releases the previous cached segmenter, runs garbage collection and clears unused CUDA allocations before constructing the next checkpoint. All segmenter calls use PyTorch inference mode, including legacy Mask R-CNN routes. The existing shared model lock still serializes switching and inference. Frozen QA/ensemble models remain loaded separately. A failed new checkpoint load leaves no stale active method.

Download trusted model weights into the persistent cache before an interactive demo. With the service stopped, the existing container can prepare the tested pool:

```sh
docker compose run --rm --no-deps pytorch python -c '
from segmentation_methods import SegmentationMethods
m = SegmentationMethods("../server-MODELS")
for name in ("cellpose.original", "instanseg.original", "maskrcnnv1COCO.original", "sam2.original-tiny", "sam2.original-small", "sam2.original-base-plus", "sam2.original-large"):
    m.get_segmentation_fun(name)
m.clear_cache()
'
```

This loads/downloads one checkpoint at a time; it does not train models or perform a GPU sizing check. Keep the cache volume. The tested seven checkpoints include four related SAM2 sizes. A request may return fewer target candidates if a method fails or its segmentation misses an explicitly supplied foreground point. Such partial results retain successful outputs and per-method errors.

The native client can explicitly mark the box centre as a foreground point. Draw the region with that centre inside the intended cell; turn the option off to use box-only matching. API callers supply `region.target_point_xy`. This option is separate from box-centre fallback and is never inferred from a reference mask.
