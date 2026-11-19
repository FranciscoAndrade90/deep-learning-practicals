"""Helpers for the score-matching practical: the mixture, measurements, plots and checks.

You do not need to read this file: the notebook explains what each function does.
"""
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

__all__ = [
    "PIS", "MUS", "set_weights", "LIM",
    "fraction_small_mode", "wasserstein", "wasserstein_large", "log_density",
    "plot_samples", "plot_field", "plot_learned_score", "plot_ncsn_scores",
    "check_sample_data", "check_true_score", "check_make_mlp", "check_langevin",
    "check_annealed_langevin", "check_dsm_loss", "check_ncsn_loss",
]


# ---------------------------------------------------------------- the mixture

PIS = torch.tensor([0.2, 0.8])                          # weights of the two components
MUS = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])          # their means


def set_weights(pis):
    """Changes the weights of the mixture, e.g. set_weights([0.05, 0.95]).
    PIS is modified in place, so the change is seen everywhere (notebook and helpers)."""
    PIS.copy_(torch.as_tensor(pis, dtype=PIS.dtype))


def fraction_small_mode(x):
    """Fraction of the samples x (n, 2) on the side of the small mode (mu_1 = (-5, -5))."""
    return (x.sum(dim=1) < 0).float().mean().item()


def wasserstein(x, y):
    """x, y: (n, 2). Exact Wasserstein distance between two sets of samples of the same size:
    the average distance moved when matching each point of x to a point of y in the best way."""
    assert len(x) == len(y), f"wasserstein needs two sets with the same number of points, got {len(x)} and {len(y)}"
    cost = torch.cdist(x, y).numpy()
    rows, cols = linear_sum_assignment(cost)            # optimal matching
    return float(cost[rows, cols].mean())


def wasserstein_large(x, y):
    """Wasserstein distance between the points of x and y on the side of the large mode.
    This measures the shape of the samples, whatever their proportions (measured by fraction_small_mode):
    over the whole plane, moving a few points from one mode to the other (a distance of 14) would dominate."""
    a, b = x[x.sum(dim=1) > 0], y[y.sum(dim=1) > 0]
    n = min(len(a), len(b))                             # same number of points on each side
    g = torch.Generator().manual_seed(0)
    a = a[torch.randperm(len(a), generator=g)[:n]]
    b = b[torch.randperm(len(b), generator=g)[:n]]
    return wasserstein(a, b)


def log_density(x, sigma=0.0):
    """x: (n, 2); sigma: a number. Returns log p_sigma(x), shape (n,).
    (p_sigma: the mixture with every component's variance inflated to 1 + sigma^2.)"""
    var = 1.0 + sigma**2
    log_terms = [torch.log(PIS[k]) - 0.5 * ((x - MUS[k]) ** 2).sum(dim=1, keepdim=True) / var
                 - math.log(2 * math.pi * var)
                 for k in range(2)]                                # two tensors of shape (n, 1)
    return torch.logsumexp(torch.cat(log_terms, dim=1), dim=1)     # (n,)


def _component_weights(x, sigma=0.0):
    """The weights w_k(x) of the two components, shape (n, 2). (A copy is given in the notebook.)"""
    var = 1.0 + sigma**2
    logits = [torch.log(PIS[k]) - 0.5 * ((x - MUS[k]) ** 2).sum(dim=1, keepdim=True) / var
              for k in range(2)]                                   # two tensors of shape (n, 1)
    return torch.softmax(torch.cat(logits, dim=1), dim=1)          # (n, 2)


def _exact_score(x, sigma=0.0):
    """The exact score of p_sigma, used by the checks (so that they do not depend on your code)."""
    var = 1.0 + sigma**2
    w1, w2 = _component_weights(x, sigma).split(1, dim=1)
    return (w1 * (MUS[0] - x) + w2 * (MUS[1] - x)) / var


def _sample_exact(n):
    k = torch.multinomial(PIS, n, replacement=True)
    return MUS[k] + torch.randn(n, 2)


# ---------------------------------------------------------------- plots

LIM = 10                                                # plotting range
_g = torch.linspace(-LIM, LIM, 200)
_GX, _GY = torch.meshgrid(_g, _g, indexing="xy")
_GRID = torch.stack([_GX.reshape(-1), _GY.reshape(-1)], dim=1)


def _density_background(ax, sigma):
    dens = log_density(_GRID, sigma).exp().reshape(_GX.shape)
    ax.contourf(_GX, _GY, dens, levels=20, cmap="Oranges")


def _frame(ax, title):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_title(title)


def plot_samples(x, ax=None, title=None, sigma=0.0):
    """Scatters the samples x (n, 2) over the density of p_sigma.
    The default title gives the fraction of samples in the small mode."""
    ax = ax or plt.gca()
    _density_background(ax, sigma)
    ax.scatter(x[:, 0], x[:, 1], s=2, c="tab:blue", alpha=0.5)
    _frame(ax, title if title is not None else f"small mode: {fraction_small_mode(x):.0%}")


def plot_field(score_fn, ax=None, title=None, sigma=0.0, n=25):
    """Draws the directions of the vector field score_fn: (n, 2) -> (n, 2), over the density of p_sigma."""
    ax = ax or plt.gca()
    g = torch.linspace(-LIM, LIM, n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    with torch.no_grad():
        s = score_fn(pts)
    s = s / (s.norm(dim=1, keepdim=True) + 1e-8)                   # show directions only
    _density_background(ax, sigma)
    ax.quiver(X, Y, s[:, 0].reshape(X.shape), s[:, 1].reshape(X.shape), width=0.004)
    _frame(ax, title)


def plot_learned_score(model, true_score, sigma, x_data):
    """Three panels: the true score of p_sigma, the learned score, and the size of the error,
    with the training data on top."""
    with torch.no_grad():
        err = (model(_GRID) - true_score(_GRID, sigma)).norm(dim=1).reshape(_GX.shape)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_field(lambda x: true_score(x, sigma), ax=axes[0], sigma=sigma, title="true score")
    plot_field(model, ax=axes[1], sigma=sigma, title="learned score")
    im = axes[2].contourf(_GX, _GY, err, levels=20, cmap="viridis")
    axes[2].scatter(x_data[:500, 0], x_data[:500, 1], s=1, c="white", alpha=0.5)
    axes[2].set_aspect("equal"); axes[2].set_title("error $\\|s_\\theta - \\nabla \\log p_\\sigma\\|$")
    plt.colorbar(im, ax=axes[2])
    plt.show()


def plot_ncsn_scores(model, true_score, sigmas, x_data):
    """One row per noise level in sigmas: the true score of p_sigma, the score learned by the
    noise-conditional model(x, sigma), and the size of the error. The error uses the SAME colour
    scale on every row, so that the levels can be compared."""
    errs = []
    with torch.no_grad():
        for s in sigmas:
            errs.append((model(_GRID, s) - true_score(_GRID, s)).norm(dim=1).reshape(_GX.shape))
    vmax = max(e.max().item() for e in errs)
    levels = torch.linspace(0, vmax, 21).tolist()
    fig, axes = plt.subplots(len(sigmas), 3, figsize=(13, 4.3 * len(sigmas)))
    for row, (s, err) in zip(axes, zip(sigmas, errs)):
        plot_field(lambda x: true_score(x, s), ax=row[0], sigma=s, title=f"true score, $\\sigma$ = {s:.1f}")
        plot_field(lambda x: model(x, s), ax=row[1], sigma=s, title=f"learned score, $\\sigma$ = {s:.1f}")
        im = row[2].contourf(_GX, _GY, err, levels=levels, cmap="viridis")
        row[2].scatter(x_data[:500, 0], x_data[:500, 1], s=1, c="white", alpha=0.5)
        row[2].set_aspect("equal"); row[2].set_title(f"error, $\\sigma$ = {s:.1f}")
        plt.colorbar(im, ax=row[2], format="%.1f")
    fig.suptitle("error $\\|s_\\theta - \\nabla \\log p_\\sigma\\|$: same colour scale on every row", y=1.0)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- checks

def check_sample_data(sample_data):
    x = sample_data(100_000)
    assert x.shape == (100_000, 2), f"expected shape (100000, 2), got {tuple(x.shape)}"
    frac = fraction_small_mode(x)
    target = PIS[0].item()
    assert abs(frac - target) < 0.01, f"expected about {target:.0%} of samples near (-5, -5), got {frac:.1%}"
    print("sample_data: OK")


def check_true_score(true_score):
    # Compare with automatic differentiation of the log-density.
    x_test = 6 * torch.randn(100, 2)
    for s in [0.0, 1.0, 10.0]:
        out = true_score(x_test, s)
        assert out.shape == (100, 2), f"expected shape (100, 2), got {tuple(out.shape)}"
        xr = x_test.clone().requires_grad_(True)
        autograd_score, = torch.autograd.grad(log_density(xr, s).sum(), xr)
        err = (autograd_score - out).abs().max().item()
        assert err < 1e-3, f"sigma = {s}: the score differs from autograd by {err:.2e}"
    print("true_score: OK")


def check_langevin(langevin):
    # The score of a standard Gaussian is -x, so Langevin should produce standard Gaussian samples.
    x = langevin(lambda x: -x, torch.zeros(20_000, 2), eps=0.01, n_steps=500)
    assert x.shape == (20_000, 2), f"expected shape (20000, 2), got {tuple(x.shape)}"
    std = x.std(dim=0)
    assert ((std - 1).abs() < 0.05).all(), f"expected standard deviations close to 1, got {std.tolist()}"
    print("langevin: OK")


def check_annealed_langevin(annealed_langevin, sigmas):
    # On a standard Gaussian p, p_sigma is Gaussian with variance 1 + sigma^2,
    # so the final samples should have standard deviation sqrt(1 + sigma_1^2).
    try:
        x, hist = annealed_langevin(lambda x, s: -x / (1 + s**2), torch.zeros(20_000, 2),
                                    sigmas, eps=0.1, n_steps=100)
    except (TypeError, AttributeError) as e:
        if "ellipsis" in str(e):
            raise AssertionError("fill in the line marked TODO: one Langevin step") from None
        raise
    assert x.shape == (20_000, 2), f"expected shape (20000, 2), got {tuple(x.shape)}"
    assert len(hist) == len(sigmas), f"expected one history entry per level, got {len(hist)}"
    assert hist[0][0] > hist[-1][0], "the noise levels should go from large to small"
    target = math.sqrt(1 + sigmas[0].item() ** 2)
    std = x.std(dim=0)
    assert ((std - target).abs() < 0.1).all(), \
        f"expected standard deviations close to {target:.2f}, got {std.tolist()}"
    print("annealed_langevin: OK")


def check_make_mlp(make_mlp):
    net = make_mlp(3)
    assert isinstance(net, nn.Sequential), f"expected an nn.Sequential, got {type(net).__name__}"
    layers = list(net)
    n_lin = sum(isinstance(m, nn.Linear) for m in layers)
    n_silu = sum(isinstance(m, nn.SiLU) for m in layers)
    assert n_lin == 4, f"expected 4 nn.Linear layers (3 hidden + 1 output), got {n_lin}"
    assert n_silu == 3, f"expected 3 nn.SiLU activations, one after each hidden layer, got {n_silu}"
    assert isinstance(layers[-1], nn.Linear), "the last layer should be nn.Linear, with no activation after it"
    out = net(torch.randn(5, 3))
    assert out.shape == (5, 2), f"with d_in=3, expected an output of shape (5, 2), got {tuple(out.shape)}"
    n_params = sum(q.numel() for q in make_mlp(2).parameters())
    assert n_params == 33_666, f"with d_in=2 and width=128, expected 33666 parameters, got {n_params}"
    assert sum(q.numel() for q in make_mlp(2, width=16).parameters()) == 626, "the width argument should be used"
    print("make_mlp: OK")


def check_dsm_loss(dsm_loss):
    # The exact score of p_sigma minimises the loss, so it must do much better than predicting 0.
    x = _sample_exact(10_000)
    loss_exact = dsm_loss(lambda x: _exact_score(x, 1.0), x, 1.0)
    loss_zero = dsm_loss(lambda x: torch.zeros_like(x), x, 1.0)
    assert loss_exact.ndim == 0, f"the loss should be a single number, got shape {tuple(loss_exact.shape)}"
    assert abs(loss_zero.item() - 2.0) < 0.1, \
        f"with a model that outputs 0, expected a loss close to 2, got {loss_zero.item():.3f}"
    assert abs(loss_exact.item() - 1.0) < 0.1, \
        f"with the exact score, expected a loss close to 1, got {loss_exact.item():.3f}"
    print("dsm_loss: OK")


def check_ncsn_loss(ncsn_loss, sigmas):
    # The exact scores must do much better than a model that predicts 0.
    x = _sample_exact(10_000)
    loss_exact = ncsn_loss(lambda x, s: _exact_score(x, s), x, sigmas)
    loss_zero = ncsn_loss(lambda x, s: torch.zeros_like(x), x, sigmas)
    assert loss_exact.ndim == 0, f"the loss should be a single number, got shape {tuple(loss_exact.shape)}"
    assert abs(loss_zero.item() - 2.0) < 0.1, \
        f"with a model that outputs 0, expected a loss close to 2, got {loss_zero.item():.3f}"
    assert loss_exact.item() < 0.6, f"with the exact scores, expected a loss below 0.6, got {loss_exact.item():.3f}"
    print("ncsn_loss: OK")
