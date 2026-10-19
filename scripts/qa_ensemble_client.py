#!/usr/bin/env python3
"""Submit one ROI to the combined remote pipeline; no local ML calculations."""
import argparse
import json
from pathlib import Path
import zipfile

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    with args.image.open("rb") as image:
        response = requests.post(args.server.rstrip("/") + "/segmentation_2D/qa_ensemble",
                                 headers={"X-API-Token": args.token_file.read_text().strip()},
                                 files={"image": (args.image.name, image, "image/tiff")},
                                 data={"metadata": json.dumps(metadata)}, timeout=args.timeout)
    response.raise_for_status()
    args.output.write_bytes(response.content)
    with zipfile.ZipFile(args.output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    print(json.dumps({"status": manifest["status"], "run_id": manifest["run_id"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
