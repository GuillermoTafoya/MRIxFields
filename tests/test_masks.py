import torch

from fieldbridge.data.masks import clean_brain_mask, fill_holes_2d, threshold_mask


def test_threshold_mask_preserves_shape_and_binary_values() -> None:
    x = torch.tensor([[[[0.0, 0.7], [1.0, -0.5]]]])

    mask = threshold_mask(x, threshold=0.5)

    assert mask.shape == x.shape
    assert set(mask.flatten().tolist()) <= {0.0, 1.0}


def test_clean_brain_mask_fills_simple_2d_hole() -> None:
    x = torch.zeros(1, 1, 7, 7)
    x[:, :, 2:5, 2:5] = 1.0
    x[:, :, 3, 3] = 0.0

    mask = clean_brain_mask(x, threshold=0.5, kernel_size=3, iterations=1)

    assert mask.shape == x.shape
    assert mask[0, 0, 3, 3].item() == 1.0
    assert set(mask.flatten().tolist()) <= {0.0, 1.0}


def test_fill_holes_2d_preserves_shape() -> None:
    mask = torch.ones(1, 1, 5, 5)
    mask[:, :, 2, 2] = 0.0

    filled = fill_holes_2d(mask)

    assert filled.shape == mask.shape
    assert filled[0, 0, 2, 2].item() == 1.0
