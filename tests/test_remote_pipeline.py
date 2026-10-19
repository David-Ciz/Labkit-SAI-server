from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from threading import RLock
import time
import zipfile

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
import tifffile

from qa_ensemble.api import create_router
from qa_ensemble.contracts import RunRequest, decode_image, target_mask
from qa_ensemble.frozen import centered_crop, paste_crop
from qa_ensemble.service import EnsembleService


def metadata(methods=None, outputs=None):
    return {"pipeline_id": "test", "methods": methods or ["a.original", "b.original"],
            "region": {"original_shape_yx": [40, 60], "origin_xy": [4, 6],
                       "target_box_xyxy": [2, 2, 8, 8], "target_point_xy": [4, 4]},
            "return": outputs if outputs is not None else ["ensemble_mask", "qa_scores"]}


class TestPipeline:
    __test__ = False
    provenance = {"pipeline_id": "test", "training_exposure": "test double; no learned claims"}

    def run(self, image, candidates, request):
        methods = list(candidates)
        return {"ensemble_mask": candidates[methods[0]], "vote_map": candidates[methods[0]].astype("float32"),
                "scores": [{"method": m, "qa_score": .8, "ranking_score": .6, "retained": True,
                            "fusion_weight": 1/len(methods), "selection_reason": "retained"} for m in methods],
                "stages": {"qa": {"status": "ok"}, "ensemble": {"status": "ok"}}}


class Methods:
    def list_avail_methods(self):
        return ["a.original", "b.original", "broken.original"]

    def get_segmentation_fun(self, method):
        def segment(image, prompt):
            if method.startswith("broken"):
                raise RuntimeError("backend failure")
            result = np.zeros(image.shape, dtype="uint16")
            result[3:7, 3:7] = 7
            result[8:10, 9:12] = 11
            return result
        return segment


def app(pipelines=None):
    def auth(x_api_token: str = Header(default="")):
        if x_api_token != "test-token":
            raise HTTPException(404)
    application = FastAPI(dependencies=[Depends(auth)])
    application.include_router(create_router(Methods(), RLock(), pipelines={"test": TestPipeline()} if pipelines is None else pipelines))
    return application


def post(client, meta, image=None, auth=True):
    buf = BytesIO()
    tifffile.imwrite(buf, np.zeros((12, 16), dtype="uint16") if image is None else image, photometric="minisblack")
    return client.post("/segmentation_2D/qa_ensemble", data={"metadata": json.dumps(meta)},
                       files={"image": ("roi.tif", buf.getvalue(), "image/tiff")},
                       headers={"X-API-Token": "test-token"} if auth else {})


def unpack(response):
    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(BytesIO(response.content))
    return archive, json.loads(archive.read("manifest.json"))


def test_combined_authenticated_request_returns_selected_outputs_and_coordinates():
    with TestClient(app()) as client:
        archive, manifest = unpack(post(client, metadata()))
        assert manifest["status"] == "ok"
        assert manifest["region"]["origin_xy"] == [4, 6]
        assert set(archive.namelist()) == {"manifest.json", "ensemble_mask.tif", "scores.csv"}
        mask = tifffile.imread(BytesIO(archive.read("ensemble_mask.tif")))
        assert mask.shape == (12, 16) and mask[4, 4] == 1 and mask[8, 9] == 0
        assert archive.read("scores.csv").decode().splitlines()[0] == "method,qa_score"


def test_all_output_choices_and_original_instance_labels():
    with TestClient(app()) as client:
        archive, manifest = unpack(post(client, metadata(outputs=["source_segmentations", "candidate_masks", "selection", "ensemble_weights", "vote_map", "diagnostics"])))
        source = manifest["files"]["source_segmentations"][0]["file"]
        assert tifffile.imread(BytesIO(archive.read(source)))[8, 9] == 11
        assert "ensemble_mask.tif" not in archive.namelist()
        assert "qa_score" not in archive.read("scores.csv").decode().splitlines()[0]


def test_partial_failure_preserves_successful_outputs():
    with TestClient(app()) as client:
        archive, manifest = unpack(post(client, metadata(methods=["a.original", "broken.original"])))
        assert manifest["status"] == "partial"
        assert manifest["methods"][1]["status"] == "failed"
        assert "ensemble_mask.tif" in archive.namelist()


def test_one_candidate_is_explicit_and_zero_candidates_has_no_mask():
    with TestClient(app()) as client:
        _, manifest = unpack(post(client, metadata(methods=["a.original"])))
        assert manifest["status"] == "single_candidate"
        archive, manifest = unpack(post(client, metadata(methods=["broken.original"])))
        assert manifest["status"] == "no_result"
        assert "ensemble_mask.tif" not in archive.namelist()


@pytest.mark.parametrize("change", [{"methods": ["a.original", "a.original"]}, {"methods": ["unknown"]},
                                    {"return": ["arbitrary"]}, {"region": {"original_shape_yx": [1, 1], "target_box_xyxy": [0, 0, 3, 3]}}])
def test_invalid_requests_are_rejected_before_segmentation(change):
    with TestClient(app()) as client:
        meta = metadata(); meta.update(change)
        assert post(client, meta).status_code == 422


def test_auth_and_missing_pipeline():
    with TestClient(app()) as client:
        assert post(client, metadata(), auth=False).status_code == 404
    with TestClient(app(pipelines={})) as client:
        assert post(client, metadata()).status_code == 503


def test_target_fallback_does_not_guess_between_cells():
    request = RunRequest.model_validate(metadata())
    labels = np.zeros((12, 16), dtype="uint16")
    labels[2, 2] = 1; labels[7, 7] = 2
    with pytest.raises(ValueError, match="Ambiguous"):
        target_mask(labels, request)
    labels[7, 7] = 0
    mask, info = target_mask(labels, request)
    assert mask[2, 2] and info["match"] == "single_instance_in_box"


def test_padded_rectangular_crop_round_trip():
    image = np.arange(35).reshape(5, 7)
    crop, box = centered_crop(image, (1, 1), 8)
    restored = paste_crop(crop, box, image.shape)
    assert crop.shape == (8, 8)
    np.testing.assert_array_equal(restored[:5, :5], image[:5, :5])
    assert np.all(restored[:, 5:] == 0)


def test_tiff_header_limit_and_nonfinite_inputs():
    buf = BytesIO(); tifffile.imwrite(buf, np.zeros((10, 10)))
    with pytest.raises(ValueError, match="pixel limit"):
        decode_image(buf.getvalue(), 99)
    buf = BytesIO(); tifffile.imwrite(buf, np.full((10, 10), np.nan))
    with pytest.raises(ValueError, match="finite"):
        decode_image(buf.getvalue(), 100)


def test_model_selection_is_serialized_across_requests():
    active, observed = [None], []
    def segment(method, image, request):
        active[0] = method; time.sleep(.01); observed.append((method, active[0]))
        return np.ones(image.shape, dtype="uint16")
    service = EnsembleService(segment, {"test": TestPipeline()}, lambda m: {}, RLock())
    def run(method):
        service.run(np.zeros((12, 16)), RunRequest.model_validate(metadata(methods=[method])))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, ["a.original", "b.original"]*3))
    assert all(expected == actual for expected, actual in observed)
