from io import BytesIO
import importlib.util
import json
from pathlib import Path
from threading import RLock
import zipfile

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
import tifffile

from qa_ensemble.api import create_router
from qa_ensemble.records import Recorder, load_records
from test_remote_pipeline import Methods, TestPipeline, metadata

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def client(recorder):
    def auth(x_api_token: str = Header(default="")):
        if x_api_token != "test-token":
            raise HTTPException(404)
    application = FastAPI(dependencies=[Depends(auth)])
    application.include_router(create_router(Methods(), RLock(), pipelines={"test": TestPipeline()}, recorder=recorder))
    return TestClient(application, headers={"X-API-Token": "test-token"})


def tiff(array):
    buf = BytesIO()
    tifffile.imwrite(buf, array)
    return buf.getvalue()


def run(c, image_id="img-a"):
    meta = metadata()
    meta["image_id"] = image_id
    r = c.post("/segmentation_2D/qa_ensemble", files={"image": ("roi.tif", tiff(np.ones((12, 14), "float32")), "image/tiff")},
               data={"metadata": json.dumps(meta)})
    assert r.status_code == 200
    return json.loads(zipfile.ZipFile(BytesIO(r.content)).read("manifest.json"))


def test_records_and_feedback_round_trip(tmp_path):
    c = client(Recorder(tmp_path))
    manifest = run(c)
    assert manifest["recorded"] is True
    run_dir = tmp_path / manifest["run_id"]
    assert (run_dir / "image.tif").exists() and len(json.loads((run_dir / "candidates.json").read_text())) == 2
    assert list(load_records(tmp_path)) == []  # no feedback yet: not a training example

    final = np.zeros((12, 14), "uint8")
    final[3:7, 3:8] = 1
    r = c.post("/segmentation_2D/qa_ensemble/feedback", data={"run_id": manifest["run_id"], "action": "edited"},
               files={"mask": ("mask.tif", tiff(final), "image/tiff")})
    assert r.status_code == 200, r.text
    (rid, request, image, mask, candidates), = load_records(tmp_path)
    assert rid == manifest["run_id"] and request["image_id"] == "img-a"
    assert mask.sum() == 20 and set(candidates) == {"a.original", "b.original"}

    r = c.post("/segmentation_2D/qa_ensemble/feedback", data={"run_id": manifest["run_id"], "action": "rejected"})
    assert r.status_code == 200 and list(load_records(tmp_path)) == []


@pytest.mark.parametrize("data,files,status", [
    ({"run_id": "../../etc", "action": "applied"}, None, 422),
    ({"run_id": "0" * 32, "action": "applied"}, {"mask": ("m.tif", tiff(np.ones((12, 14), "uint8")), "image/tiff")}, 404),
    ({"run_id": "0" * 32, "action": "bogus"}, None, 422),
])
def test_feedback_rejects_bad_input(tmp_path, data, files, status):
    assert client(Recorder(tmp_path)).post("/segmentation_2D/qa_ensemble/feedback", data=data, files=files).status_code == status


def test_feedback_checks_mask_shape(tmp_path):
    c = client(Recorder(tmp_path))
    rid = run(c)["run_id"]
    r = c.post("/segmentation_2D/qa_ensemble/feedback", data={"run_id": rid},
               files={"mask": ("m.tif", tiff(np.ones((5, 5), "uint8")), "image/tiff")})
    assert r.status_code == 422 and "shape" in r.text


def test_recording_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LABKIT_RECORD_DIR", raising=False)
    c = client(None)
    assert run(c)["recorded"] is False
    assert c.get("/segmentation_2D/qa_ensemble/capabilities").json()["recording"] is False
    assert c.post("/segmentation_2D/qa_ensemble/feedback", data={"run_id": "0" * 32}).status_code == 409


def test_trainer_reads_records_in_pipeline_crop(tmp_path):
    trainer = load("train_pool_bundle", "scripts/train_pool_bundle.py")
    c = client(Recorder(tmp_path))
    for image_id in ("img-a", "img-a", "img-b"):
        rid = run(c, image_id)["run_id"]
        final = np.zeros((12, 14), "uint8")
        final[3:7, 3:7] = 1
        c.post("/segmentation_2D/qa_ensemble/feedback", data={"run_id": rid},
               files={"mask": ("m.tif", tiff(final), "image/tiff")})
    cells, raws, refs, masks, status = trainer.collect_records(tmp_path, ["a.original", "b.original"], min_area=4)
    assert len(cells) == 3 and cells.group.nunique() == 2
    cid = cells.cell_id[0]
    # Anchor (4, 4) sits at the crop centre; the 6x6 box shifts with it.
    assert cells.point[0] == [128, 128] and cells.box[0] == [126, 126, 132, 132]
    assert refs[cid].shape == (256, 256) and refs[cid].sum() == 16
    assert masks[(cid, "a.original")].sum() == 16 and (status.status == "ok").all()
    with pytest.raises(SystemExit):
        trainer.collect_records(tmp_path, ["a.original", "not-in-pool"], min_area=4)


def test_segment_prompt_and_roi():
    job = load("labkit_job", "deploy/lexis/labkit_job.py")
    mask = np.zeros((50, 60), bool)
    mask[10:30, 20:40] = True
    box, point = job.proposal_prompt(mask)
    assert box[0] < 20 and box[2] > 40 and mask[point[1], point[0]]
    assert job.roi_around((1000, 1000), 5, 990) == (0, 744, 256, 1000)
    assert job.roi_around((100, 120), 60, 50) == (0, 0, 120, 100)


def test_warm_up_loads_the_pool_before_serving(monkeypatch, capsys):
    class Pipeline(TestPipeline):
        provenance = {"pipeline_id": "test", "training_exposure": {"methods": ["a.original", "broken.original"]}}
    monkeypatch.setenv("LABKIT_REQUIRE_PIPELINE", "test")
    monkeypatch.setenv("LABKIT_WARMUP", "1")
    application = FastAPI()
    application.include_router(create_router(Methods(), RLock(), pipelines={"test": Pipeline()}))
    with TestClient(application):
        out = capsys.readouterr().out
    assert "warm-up a.original" in out and "warm-up broken.original failed" in out


def test_required_pipeline_missing_fails_startup(monkeypatch):
    monkeypatch.setenv("LABKIT_REQUIRE_PIPELINE", "absent")
    application = FastAPI()
    application.include_router(create_router(Methods(), RLock(), pipelines={"test": TestPipeline()}))
    with pytest.raises(RuntimeError, match="absent"):
        with TestClient(application):
            pass


def test_extra_models_dir_adds_fine_tuned_methods(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / "server-pytorch"))
    monkeypatch.chdir(ROOT / "server-pytorch")  # the module resolves its wrappers relative to here
    import segmentation_methods as nets
    (tmp_path / "base").mkdir(); (tmp_path / "extra").mkdir()
    (tmp_path / "base" / "original.cellpose.torch.model").touch()
    (tmp_path / "extra" / "pool-ft.cellpose.torch.model").touch()
    monkeypatch.setenv("LABKIT_EXTRA_MODELS_DIR", str(tmp_path / "extra"))
    methods = nets.SegmentationMethods(str(tmp_path / "base"))
    assert methods.available_methods["cellpose.pool-ft"] == str(tmp_path / "extra" / "pool-ft.cellpose.torch.model")
    assert "cellpose.original" in methods.available_methods


def test_trainer_groups_per_record_when_client_gives_no_image_id(tmp_path):
    trainer = load("train_pool_bundle", "scripts/train_pool_bundle.py")
    c = client(Recorder(tmp_path))
    for _ in range(3):
        rid = run(c, "labkit-captured-view")["run_id"]
        final = np.zeros((12, 14), "uint8"); final[3:7, 3:7] = 1
        c.post("/segmentation_2D/qa_ensemble/feedback", data={"run_id": rid}, files={"mask": ("m.tif", tiff(final), "image/tiff")})
    cells, *_ = trainer.collect_records(tmp_path, ["a.original", "b.original"], min_area=4)
    assert cells.group.nunique() == 3
