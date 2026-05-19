import torch

from sedd.models import DilatedConvNet
from sedd.noise import beta_to_tau, corrupt_spins, ell_to_tau, make_level_sampler, sedd_target_ratio, tau_to_ell
from sedd.training import denoiser_loss, sedd_loss
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
    sampler = make_level_sampler("ell", -2.0, 2.0, output_kind="tau")
    levels = sampler(16)
    assert levels.shape == (16, 1)
    assert torch.all((levels > 0.0) & (levels < 1.0))


def test_empirical_exact_validation_helpers():
    clean = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    x = torch.tensor([[1.0, -1.0]])
    mean = empirical_posterior_mean(clean, x, tau=0.0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    log_ratios = empirical_noisy_log_ratios(clean, x, tau=0.0)
    assert torch.allclose(log_ratios, torch.zeros_like(log_ratios), atol=1e-6)
