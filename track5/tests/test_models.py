import numpy as np
import torch
from PIL import Image

from track5.models import AverageEnsemble, build_model
from track5.models.preprocess import eval_crop, to_tensor, train_crop

STUB_CFG = {"model": {"stub": True, "head": "linear", "pool": "cls",
                      "pretrained": False, "freeze_backbone": False}}


def make_img(w, h, seed=1):
    rng = np.random.Generator(np.random.PCG64(seed))
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def test_stub_forward_shape():
    model = build_model(STUB_CFG)
    z = model(torch.randn(2, 3, 448, 448))
    assert z.shape == (2,)
    assert torch.isfinite(z).all()


def test_freeze_backbone_leaves_head_only():
    cfg = {"model": {**STUB_CFG["model"], "freeze_backbone": True}}
    model = build_model(cfg)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("head.") for n in trainable)
    model.train()
    assert not model.backbone.training  # frozen backbone stays in eval mode


def test_crops_sizes():
    rng = np.random.Generator(np.random.PCG64(0))
    for w, h in [(600, 500), (300, 200), (448, 448), (100, 700)]:
        img = make_img(w, h)
        assert train_crop(img, rng, 448).size == (448, 448)
        assert eval_crop(img, 448).size == (448, 448)


def test_train_crop_determinism_and_variation():
    img = make_img(800, 600)
    a = np.asarray(train_crop(img, np.random.Generator(np.random.PCG64(5)), 448))
    b = np.asarray(train_crop(img, np.random.Generator(np.random.PCG64(5)), 448))
    c = np.asarray(train_crop(img, np.random.Generator(np.random.PCG64(6)), 448))
    assert (a == b).all()
    assert not (a == c).all()


def test_eval_crop_deterministic_center():
    img = make_img(800, 600)
    a = np.asarray(eval_crop(img, 448))
    b = np.asarray(eval_crop(img, 448))
    assert (a == b).all()


def test_no_resize_native_crop():
    # a 2x2 checkerboard pattern upscale would blur under resize; crop must
    # preserve exact source pixels for large-enough images
    arr = np.zeros((500, 500, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    img = Image.fromarray(arr)
    out = np.asarray(eval_crop(img, 448))
    y0 = (500 - 448) // 2
    assert (out == arr[y0:y0 + 448, y0:y0 + 448]).all()


def test_to_tensor_normalization():
    img = make_img(32, 32)
    t = to_tensor(img)
    assert t.shape == (3, 32, 32)
    assert t.dtype == torch.float32


def test_average_ensemble():
    m1, m2 = build_model(STUB_CFG), build_model(STUB_CFG)
    cals = [{"temperature": 2.0, "alpha": 0.5}, {"temperature": 1.0, "alpha": 0.0}]
    ens = AverageEnsemble([m1, m2], cals)
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        p = ens(x)
        manual = (torch.sigmoid((m1(x) + 0.5) / 2.0) + torch.sigmoid(m2(x))) / 2
    assert p.shape == (3,)
    assert ((p >= 0) & (p <= 1)).all()
    assert torch.allclose(p, manual)
