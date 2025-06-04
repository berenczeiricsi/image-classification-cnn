import numpy as np
import pytest
import types
import sys
import importlib


def get_dataset_module():
    if 'dataset' in sys.modules:
        return sys.modules['dataset']
    torch_mock = types.ModuleType('torch')
    torch_mock.utils = types.ModuleType('torch.utils')
    torch_mock.utils.data = types.ModuleType('torch.utils.data')
    class DummyDataset:
        pass
    torch_mock.utils.data.Dataset = DummyDataset
    sys.modules.setdefault('torch', torch_mock)
    sys.modules.setdefault('torch.utils', torch_mock.utils)
    sys.modules.setdefault('torch.utils.data', torch_mock.utils.data)
    torchvision_mock = types.ModuleType('torchvision')
    torchvision_mock.transforms = types.ModuleType('torchvision.transforms')
    class DummyCompose:
        pass
    torchvision_mock.transforms.Compose = DummyCompose
    sys.modules.setdefault('torchvision', torchvision_mock)
    sys.modules.setdefault('torchvision.transforms', torchvision_mock.transforms)
    return importlib.import_module('dataset')


def test_prepare_image_shapes():
    dataset = get_dataset_module()
    image = np.zeros((1, 40, 50), dtype=np.uint8)
    resized, subarea = dataset.prepare_image(image, width=64, height=64, x=16, y=16, size=32)
    assert resized.shape == (1, 64, 64)
    assert subarea.shape == (1, 32, 32)


@pytest.mark.parametrize(
    "param,value",
    [
        ("image", np.zeros((40, 50), dtype=np.uint8)),
        ("width", 30),
        ("height", 30),
        ("size", 30),
    ],
)
def test_prepare_image_invalid_params(param, value):
    dataset = get_dataset_module()
    image = np.zeros((1, 40, 50), dtype=np.uint8)
    kwargs = dict(width=64, height=64, x=16, y=16, size=32)
    if param == "image":
        with pytest.raises(ValueError):
            dataset.prepare_image(value, **kwargs)
    else:
        kwargs[param] = value
        with pytest.raises(ValueError):
            dataset.prepare_image(image, **kwargs)


def test_prepare_image_invalid_position():
    dataset = get_dataset_module()
    image = np.zeros((1, 40, 50), dtype=np.uint8)
    with pytest.raises(ValueError):
        dataset.prepare_image(image, width=64, height=64, x=-1, y=16, size=32)
    with pytest.raises(ValueError):
        dataset.prepare_image(image, width=64, height=64, x=40, y=16, size=32)
    with pytest.raises(ValueError):
        dataset.prepare_image(image, width=64, height=64, x=16, y=-1, size=32)
    with pytest.raises(ValueError):
        dataset.prepare_image(image, width=64, height=64, x=16, y=40, size=32)
