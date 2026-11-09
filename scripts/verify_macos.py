"""One real authenticated request against the native server; preserve returned masks."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import requests
import tifffile

root = Path(__file__).resolve().parents[1]
vault = root.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8003)
parser.add_argument("--methods", nargs="+", default=["cellpose.original"])
parser.add_argument("--tag", default="cellpose-mps")
parser.add_argument("--pipeline", help="Pipeline ID; default is the first advertised bundle")
parser.add_argument("--key-file", type=Path, help="Optional token file instead of the native localhost test key")
args = parser.parse_args()
out = vault / "labkit-evaluation/native-macos" / args.tag
out.mkdir(parents=True, exist_ok=True)
url = f"http://127.0.0.1:{args.port}"
headers = {"X-API-Token": args.key_file.read_text().strip() if args.key_file else os.getenv("LABKIT_LOCAL_TEST_KEY", "1234")}
assert requests.get(url + "/runtime", timeout=10).status_code == 404
runtime = requests.get(url + "/runtime", headers=headers, timeout=15)
runtime.raise_for_status()
(out / "runtime.json").write_text(json.dumps(runtime.json(), indent=2))
caps = requests.get(url + "/segmentation_2D/qa_ensemble/capabilities", headers=headers, timeout=30)
caps.raise_for_status()
capabilities = caps.json()
(out / "capabilities.json").write_text(json.dumps(capabilities, indent=2))
assert capabilities["pipelines"], capabilities["unavailable_bundles"]
pipeline_id = args.pipeline or capabilities["pipelines"][0]["pipeline_id"]
assert pipeline_id in {p["pipeline_id"] for p in capabilities["pipelines"]}, pipeline_id
image = tifffile.imread(vault / "labkit-evaluation/images/case-00.tif").astype(np.float32)
buffer = BytesIO()
tifffile.imwrite(buffer, image, photometric="minisblack")
metadata = {"schema_version": 1, "image_id": "case-00-native-macos", "coordinate_space": "source_image",
            "pipeline_id": pipeline_id, "methods": args.methods,
            "region": {"original_shape_yx": list(image.shape), "target_box_xyxy": [80, 80, 176, 176],
                       "target_point_xy": [128, 128]},
            "return": ["ensemble_mask", "source_segmentations", "candidate_masks", "qa_scores",
                       "selection", "ensemble_weights", "vote_map", "diagnostics"]}
print("Requesting:", args.methods, "runtime:", runtime.json(), flush=True)
started = perf_counter()
response = requests.post(url + "/segmentation_2D/qa_ensemble", headers=headers,
                         files={"image": ("case-00.tif", buffer.getvalue(), "image/tiff")},
                         data={"metadata": json.dumps(metadata)}, timeout=600)
response.raise_for_status()
(out / "result.zip").write_bytes(response.content)
with zipfile.ZipFile(BytesIO(response.content)) as archive:
    manifest = json.loads(archive.read("manifest.json"))
    assert manifest["pipeline"]["pipeline_id"] == pipeline_id
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = {"client_seconds": perf_counter() - started, "status": manifest["status"],
               "pipeline_id": pipeline_id, "candidate_grouping_version": manifest["pipeline"].get("candidate_grouping_version", 1),
               "methods": manifest["methods"], "stages": manifest["stages"]}
    mask_file = manifest["files"].get("ensemble_mask")
    if mask_file:
        data = archive.read(mask_file)
        (out / "ensemble-mask.tif").write_bytes(data)
        mask = tifffile.imread(BytesIO(data)) > 0
        assert mask.shape == image.shape and mask.any()
        reference = tifffile.imread(vault / "labkit-evaluation/references/case-00-reference.tif") > 0
        summary.update(foreground_pixels=int(mask.sum()), reference_iou=float((mask & reference).sum() / (mask | reference).sum()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
assert mask_file, "No result mask; inspect saved method/stage failures"
assert manifest["stages"]["ensemble"]["status"] == "ok", manifest["stages"]
