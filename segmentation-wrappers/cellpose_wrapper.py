import torch
from cellpose import models

from qa_ensemble.runtime_device import model_device
device = model_device("cellpose")
print(f"Will use: {device}")


def create_official_model(pretrained_model = 'cpsam'):
    print(f"Creating CPv4, pretrained with {pretrained_model}")
    return models.CellposeModel(device=device, pretrained_model=pretrained_model)


def apply_model(model, img):
    masks,_,_ = model.eval([img], normalize=True, do_3D=False)
    return masks[0]


def load_model(filepath):
    """
    Here, no 'model' is explicitly provided because the loading creates new one,
    which is returned. If 'filepath' doesn't seem to work, try to use either
    an absolute path or prefix the relative path with 'models/'.
    """
    return models.CellposeModel(pretrained_model=filepath, device=device)

