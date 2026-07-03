import torch

from fieldbridge.data.domains import Domain
from fieldbridge.training.pseudo_pairs import make_pseudo_pair


def test_make_pseudo_pair_preserves_shapes_and_target() -> None:
    torch.manual_seed(4)
    x_high = torch.randn(2, 1, 16, 16)
    high_domain = Domain(7.0, "T2-FLAIR")
    low_domain = Domain(0.1, "T2-FLAIR")

    x_low, target = make_pseudo_pair(
        x_high,
        high_domain,
        low_domain,
        generator=torch.Generator().manual_seed(8),
    )

    assert x_low.shape == x_high.shape
    assert target.shape == x_high.shape
    assert torch.equal(target, x_high)
    assert not torch.allclose(x_low, x_high)
    assert torch.isfinite(x_low).all()


def test_make_pseudo_pair_accepts_domain_batches() -> None:
    x_high = torch.randn(2, 1, 8, 12, 12)
    high_domains = [Domain(7.0, "T1w"), Domain(3.0, "T2w")]
    low_domains = [Domain(0.1, "T1w"), Domain(1.5, "T2w")]

    x_low, target = make_pseudo_pair(
        x_high,
        high_domains,
        low_domains,
        generator=torch.Generator().manual_seed(2),
    )

    assert x_low.shape == x_high.shape
    assert torch.equal(target, x_high)
