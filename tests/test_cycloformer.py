"""Smoke tests for CycloFormerModule forward pass and gradient flow."""

import torch
import pytest


def _make_cpmp_batch(B=2, N_conf=3, N_atoms=6, F=25):
    """Build a minimal synthetic CPMP-format batch dict."""
    return {
        "node_feat": torch.randn(B, N_conf, N_atoms, F),
        "adj": torch.eye(N_atoms).unsqueeze(0).unsqueeze(0).expand(B, N_conf, -1, -1),
        "dist": torch.rand(B, N_conf, N_atoms, N_atoms),
        "atom_mask": torch.ones(B, N_atoms, dtype=torch.bool),
        "conformer_mask": torch.ones(B, N_conf, dtype=torch.bool),
        "target": torch.randn(B, 1),
    }


def _make_egnn_batch(B=2, N_conf=3, N_atoms=6, F=25):
    """Build a minimal synthetic EGNN-format batch dict."""
    return {
        "node_feat": torch.randn(B, N_conf, N_atoms, F),
        "coords": torch.randn(B, N_conf, N_atoms, 3),
        "conformer_mask": torch.ones(B, N_conf, dtype=torch.bool),
        "target": torch.randn(B, 1),
    }


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_cycloformer_cpmp_forward_shape(pooling):
    from src.networks.cycloformer_model import CycloFormerModule

    d_atom = 25
    d_gnn = 32
    d_model = 32

    module = CycloFormerModule(
        gnn_type="cpmp",
        d_atom=d_atom,
        d_gnn=d_gnn,
        d_model=d_model,
        n_tf_heads=2,
        n_tf_layers=1,
        d_ff=64,
        dropout=0.0,
        pooling=pooling,
        max_conformers=8,
        device=torch.device("cpu"),
        local_rank=0,
        N=1,
        h=2,
        dropout_cpmp=0.0,
        N_dense=1,
    )

    batch = _make_cpmp_batch(B=2, N_conf=3, N_atoms=6, F=d_atom)
    pred = module.forward(batch)
    assert pred.shape == (2, 1), f"Expected (2,1) got {pred.shape}"


def test_cycloformer_cpmp_gradient_flow():
    from src.networks.cycloformer_model import CycloFormerModule

    d_atom = 25
    module = CycloFormerModule(
        gnn_type="cpmp",
        d_atom=d_atom,
        d_gnn=32,
        d_model=32,
        n_tf_heads=2,
        n_tf_layers=1,
        d_ff=64,
        dropout=0.0,
        pooling="cls",
        max_conformers=8,
        device=torch.device("cpu"),
        local_rank=0,
        N=1,
        h=2,
        N_dense=1,
    )

    batch = _make_cpmp_batch(B=2, N_conf=2, N_atoms=5, F=d_atom)
    target = batch["target"]
    pred = module.forward(batch)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    # Check that at least one model parameter has a gradient
    has_grad = any(p.grad is not None for p in module.model.parameters())
    assert has_grad, "No gradients found after backward pass"


def test_cycloformer_conformer_mask():
    """Padding conformers (mask=False) should not crash the forward pass."""
    from src.networks.cycloformer_model import CycloFormerModule

    d_atom = 25
    module = CycloFormerModule(
        gnn_type="cpmp",
        d_atom=d_atom,
        d_gnn=32,
        d_model=32,
        n_tf_heads=2,
        n_tf_layers=1,
        d_ff=64,
        dropout=0.0,
        pooling="cls",
        max_conformers=8,
        device=torch.device("cpu"),
        local_rank=0,
        N=1,
        h=2,
        N_dense=1,
    )

    B, N_conf, N_atoms = 2, 4, 5
    batch = _make_cpmp_batch(B=B, N_conf=N_conf, N_atoms=N_atoms, F=d_atom)
    # Mask out last 2 conformers for molecule 0
    batch["conformer_mask"][0, 2:] = False

    pred = module.forward(batch)
    assert pred.shape == (B, 1)
    assert torch.isfinite(pred).all()
