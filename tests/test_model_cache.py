"""Model lifecycle tests use weak references, not heavyweight checkpoints."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch


@pytest.fixture
def registry(monkeypatch):
    wrappers = {name: SimpleNamespace() for name in
                ['cellpose_wrapper', 'instanseg_wrapper', 'maskrcnn_wrapper', 'sam2_demo']}
    for name, module in wrappers.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'utils', SimpleNamespace(list_models_files=lambda *args: []))
    path = Path(__file__).parents[1] / 'server-pytorch/segmentation_methods.py'
    spec = importlib.util.spec_from_file_location('model_cache_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.SegmentationMethods()
    instance.available_methods.update({'cellpose.original': 'unused', 'maskrcnnv1COCO.original': 'unused'})
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: False)
    return module, instance


class Model:
    pass


def test_old_checkpoint_is_released_before_next_allocation(registry):
    module, methods = registry
    refs = []
    def load_first():
        model = Model(); refs.append(weakref.ref(model)); return model
    module.C.create_official_model = load_first
    module.C.apply_model = lambda model, image: image
    methods.get_segmentation_fun('cellpose.original')(None, None)
    assert refs[0]() is not None
    def load_next():
        assert refs[0]() is None, 'Old checkpoint still alive while allocating next checkpoint'
        return Model()
    module.M.create_official_model = load_next
    module.M.apply_model = lambda model, image: (image, {})
    module.utils.basic_inplace_normalization = lambda image: image
    methods.get_segmentation_fun('maskrcnnv1COCO.original')(None, None)


def test_same_checkpoint_is_reused_and_inference_mode_covers_legacy_wrappers(registry):
    module, methods = registry
    loaded = []
    module.C.create_official_model = lambda: loaded.append(Model()) or loaded[-1]
    module.C.apply_model = lambda model, image: (torch.is_inference_mode_enabled(), torch.is_grad_enabled())
    first = methods.get_segmentation_fun('cellpose.original')
    assert first is methods.get_segmentation_fun('cellpose.original')
    assert len(loaded) == 1
    assert first(None, None) == (True, False)
    assert not torch.is_inference_mode_enabled()


def test_load_failure_does_not_leave_a_stale_active_method(registry):
    module, methods = registry
    module.C.create_official_model = lambda: Model()
    module.C.apply_model = lambda model, image: image
    methods.get_segmentation_fun('cellpose.original')
    def fail():
        raise RuntimeError('load failed')
    module.M.create_official_model = fail
    with pytest.raises(RuntimeError, match='load failed'):
        methods.get_segmentation_fun('maskrcnnv1COCO.original')
    assert methods.last_used_fun is None
    assert methods.last_used_method not in methods.available_methods


def test_unknown_method_does_not_evict_working_cache(registry):
    module, methods = registry
    module.C.create_official_model = lambda: Model()
    module.C.apply_model = lambda model, image: image
    function = methods.get_segmentation_fun('cellpose.original')
    assert methods.get_segmentation_fun('missing.original') is None
    assert methods.get_segmentation_fun('cellpose.original') is function


def test_gpu_allocator_is_cleared_before_loading_next_model(registry, monkeypatch):
    module, methods = registry
    events = []
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(module.torch.cuda, 'empty_cache', lambda: events.append('release'))
    module.C.create_official_model = lambda: events.append('allocate') or Model()
    module.C.apply_model = lambda model, image: image
    methods.get_segmentation_fun('cellpose.original')
    assert events == ['release', 'allocate']
