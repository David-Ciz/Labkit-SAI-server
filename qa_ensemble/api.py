from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
from threading import RLock
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import ValidationError
from starlette.responses import Response
import numpy as np
import requests
import tifffile

from .contracts import OUTPUTS, RunRequest, decode_image
from .service import EnsembleService


def create_router(methods, lock: RLock, pipelines=None):
    """Mount in the authenticated PyTorch app; inject artifacts/backends in tests."""
    registry = pipelines if pipelines is not None else {}
    failures = []
    max_pixels = int(os.getenv("LABKIT_QA_MAX_PIXELS", "4194304"))
    max_bytes = int(os.getenv("LABKIT_QA_MAX_BYTES", "67108864"))
    max_methods = int(os.getenv("LABKIT_QA_MAX_METHODS", "8"))
    tf_url = os.getenv("LABKIT_TENSORFLOW_URL", "").rstrip("/")
    tf_token_file = os.getenv("LABKIT_TENSORFLOW_TOKEN_FILE", "")
    tf_timeout = float(os.getenv("LABKIT_TENSORFLOW_TIMEOUT", "180"))

    def tf_headers():
        if not tf_url or not tf_token_file:
            raise ValueError("StarDist backend is not configured")
        token = Path(tf_token_file).read_text().strip()
        if not token:
            raise ValueError("StarDist backend token is empty")
        return {"X-API-Token": token}

    def available_methods():
        with lock:
            identities = list(methods.list_avail_methods())
        if tf_url:
            # Known model advertisements, not a network request on every discovery.
            model_root = Path(__file__).resolve().parents[1] / "server-MODELS"
            identities += ["stardist." + p.name.split(".")[0] for p in model_root.glob("*.stardist.tensorflow.model")]
        return sorted(set(identities))

    def info(method):
        return {"method_id": method, "backend": "tensorflow" if method.startswith("stardist.") else "pytorch",
                "prompt": "box" if method.startswith("sam2.") else "crop"}

    def segment(method, image, request):
        if method.startswith("stardist."):
            data = BytesIO()
            tifffile.imwrite(data, image, photometric="minisblack")
            response = requests.post(tf_url + "/segmentation_2D/on_posted_tiff/use/" + quote(method, safe=""),
                                     data=data.getvalue(), headers={**tf_headers(), "Content-Type": "application/octet-stream"},
                                     timeout=tf_timeout)
            response.raise_for_status()
            return decode_image(response.content, max_pixels)
        fun = methods.get_segmentation_fun(method)
        if fun is None:
            raise ValueError("Method is unavailable")
        x0, y0, x1, y1 = request.region.target_box_xyxy
        return fun(image, (x0, y0, x1-1, y1-1))

    service = EnsembleService(segment, registry, info, lock)

    @asynccontextmanager
    async def lifespan(app):
        if pipelines is None:
            from .frozen import load_pipelines
            root = os.getenv("LABKIT_QA_BUNDLES", str(Path(__file__).resolve().parents[1] / "qa-bundles"))
            loaded, unavailable = load_pipelines(root, os.getenv("LABKIT_QA_DEVICE", "cpu"))
            registry.update(loaded)
            failures.extend(unavailable)
        yield

    router = APIRouter(lifespan=lifespan)

    @router.get("/segmentation_2D/qa_ensemble/capabilities")
    def capabilities():
        return {"schema_version": 1, "target_scope": "one_cell", "methods": [info(m) for m in available_methods()],
                "pipelines": [p.provenance for p in registry.values()], "unavailable_bundles": failures,
                "outputs": OUTPUTS, "limits": {"max_pixels": max_pixels, "max_bytes": max_bytes, "max_methods": max_methods}}

    @router.post("/segmentation_2D/qa_ensemble")
    def run(image: UploadFile, metadata: str = Form()):
        if len(metadata.encode()) > 65536:
            raise HTTPException(413, "Request metadata is too large")
        try:
            request = RunRequest.model_validate_json(metadata)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if request.pipeline_id not in registry:
            raise HTTPException(503, "Requested frozen pipeline is unavailable; install its compatible inference bundle")
        if len(request.methods) > max_methods:
            raise HTTPException(422, "Too many selected methods")
        unknown = set(request.methods) - set(available_methods())
        if unknown:
            raise HTTPException(422, "Unknown methods: " + ", ".join(sorted(unknown)))
        payload = image.file.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise HTTPException(413, "ROI upload exceeds the configured byte limit")
        try:
            raw = decode_image(payload, max_pixels)
            request.validate_image(raw.shape)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        result = service.run(raw, request)
        return Response(result, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="labkit-qa-result.zip"'})

    return router
