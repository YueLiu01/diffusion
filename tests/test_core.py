import torch
from torch.utils.data import DataLoader

from sedd.checkpoint import load_checkpoint, save_checkpoint
from sedd.models import DilatedConvNet
from sedd.noise import beta_to_tau, corrupt_spins, ell_to_tau, level_to_tau, make_level_sampler, sedd_target_ratio, tau_to_ell
from sedd.training import denoiser_loss, sedd_loss, train_epochs_with_validation
from sedd.validation import empirical_noisy_log_ratios, empirical_posterior_mean


def test_binary_channel_shapes_and_values():
    z = torch.tensor([[1.0, -1.0, 1.0]])
    tau = torch.tensor([[0.5]])
    x = corrupt_spins(z, tau)
    assert x.shape == z.shape
    assert set(torch.unique(x).tolist()).issubset({-1.0, 1.0})
    ratio = sedd_target_ratio(z, z, tau)
    assert torch.allclose(ratio, torch.full_like(z, 1.0 / 3.0))


def test_model_and_losses_have_expected_shapes():
    torch.manual_seed(0)
    z = torch.sign(torch.randn(4, 10))
    beta = torch.full((4, 1), 0.2)
    model = DilatedConvNet(length=10, hidden_channels=8, level_embedding_dim=8)
    out = model(z, beta)
    assert out.shape == z.shape
    loss = sedd_loss(model, z, beta, augment_z2=False, augment_shift=False)
    assert torch.isfinite(loss)
    denoiser = DilatedConvNet(length=10, hidden_channels=8, level_embedding_dim=8, output_activation="tanh")
    mse = denoiser_loss(denoiser, z, beta, augment_z2=False, augment_shift=False)
    assert torch.isfinite(mse)


def test_beta_to_tau():
    beta = torch.tensor([[0.0], [0.5]])
    tau = beta_to_tau(beta)
    assert tau[0].item() == 0.0
    assert 0.0 < tau[1].item() < 1.0


def test_ell_sampler_round_trip_and_output_kind():
    tau = torch.tensor([[0.2], [0.8]])
    assert torch.allclose(ell_to_tau(tau_to_ell(tau)), tau, atol=1e-6)
    sampler = make_level_sampler("ell", 0.01, 2.0, output_kind="tau")
    levels = sampler(16)
    assert levels.shape == (16, 1)
    assert torch.all((levels > 0.0) & (levels < 1.0))
    ell_sampler = make_level_sampler("ell", 0.01, 2.0, output_kind="ell")
    ell = ell_sampler(16)
    assert torch.all((ell >= 0.01) & (ell <= 2.0))
    assert torch.allclose(level_to_tau(ell, "ell"), ell_to_tau(ell))


def test_empirical_exact_validation_helpers():
    clean = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    x = torch.tensor([[1.0, -1.0]])
    mean = empirical_posterior_mean(clean, x, tau=0.0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    log_ratios = empirical_noisy_log_ratios(clean, x, tau=0.0)
    assert torch.allclose(log_ratios, torch.zeros_like(log_ratios), atol=1e-6)


def test_train_epochs_with_validation_returns_epoch_metrics():
    torch.manual_seed(0)
    data = torch.sign(torch.randn(8, 6))
    train_loader = DataLoader(data[:6], batch_size=2)
    val_loader = DataLoader(data[6:], batch_size=2)
    model = DilatedConvNet(length=6, hidden_channels=8, level_embedding_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def sampler(batch_size, device=None):
        return torch.full((batch_size, 1), 0.2, device=device)

    history = train_epochs_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        sampler,
        epochs=1,
        device=torch.device("cpu"),
        objective="sedd",
    )
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert history[0]["train_steps"] == 3
    assert torch.isfinite(torch.tensor(history[0]["train_loss"]))
    assert torch.isfinite(torch.tensor(history[0]["val_loss"]))


def test_checkpoint_sanitizes_path_config(tmp_path):
    model = DilatedConvNet(length=4, hidden_channels=4, level_embedding_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, {"output": tmp_path})
    checkpoint = load_checkpoint(path, map_location="cpu")
    assert checkpoint["config"]["output"] == str(tmp_path)
