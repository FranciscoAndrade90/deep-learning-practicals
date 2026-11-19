"""Helpers for the GAN practical: data, measurements, plots and checks.

You do not need to read this file: the notebook explains what each function does.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

__all__ = [
    "PIS", "MUS", "LIM", "sample_data", "plot_real", "real_1d", "fake_1d",
    "fraction_small_mode", "wasserstein", "wasserstein_large",
    "X_EVAL", "W_FLOOR", "W_FLOOR_LARGE", "print_floors",
    "new_history", "record", "plot_snapshots", "plot_curves",
    "plot_1d", "plot_1d_critic", "compare_lipschitz",
    "check_networks", "check_gan_losses", "check_wgan_losses",
    "check_power_iteration", "check_SNLinear", "check_max_grad_norm",
    "plot_toy_gan", "check_train_toy_gan",
]


# ---------------------------------------------------------------- data

PIS = torch.tensor([0.2, 0.8])                          # weights of the two modes
MUS = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])          # their centres
LIM = 9                                                 # plotting range


def sample_data(n=10_000, generator=None):
    """Returns n samples from the mixture, shape (n, 2)."""
    k = torch.multinomial(PIS, n, replacement=True, generator=generator)
    return MUS[k] + torch.randn(n, 2, generator=generator)


def plot_real(real_samples, ax):
    """Scatters the real samples on ax."""
    r = real_samples.numpy()
    ax.scatter(r[:, 0], r[:, 1], s=8, alpha=0.5, color="#4477AA")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.grid(True, linewidth=0.4, color="#cccccc")
    ax.set_aspect("equal")


# 1-D example of Sections 2 and 3: real samples around -2, fake samples around +2
real_1d = lambda n: -2 + 0.2 * torch.randn(n, 1)
fake_1d = lambda n: 2 + 0.2 * torch.randn(n, 1)


# ---------------------------------------------------------------- measurements

def fraction_small_mode(x):
    """x: (n, 2). Fraction of samples on the side of the small mode (the line x1 + x2 = 0 separates the modes)."""
    return (x.sum(dim=1) < 0).float().mean().item()


def wasserstein(x, y):
    """x, y: (n, 2). Exact Wasserstein-1 distance between two point clouds of the same size."""
    assert len(x) == len(y), f"wasserstein needs two sets with the same number of points, got {len(x)} and {len(y)}"
    cost = torch.cdist(x, y).numpy()
    rows, cols = linear_sum_assignment(cost)            # optimal matching
    return float(cost[rows, cols].mean())


def wasserstein_large(x_real, x_gen):
    """Wasserstein distance between the points of x_real and x_gen on the side of the large mode."""
    a, b = x_real[x_real.sum(dim=1) > 0], x_gen[x_gen.sum(dim=1) > 0]
    n = min(len(a), len(b))                             # same number of points on each side
    if n == 0:                                          # no generated point on that side: undefined
        return float("nan")
    g = torch.Generator().manual_seed(0)
    a = a[torch.randperm(len(a), generator=g)[:n]]
    b = b[torch.randperm(len(b), generator=g)[:n]]
    return wasserstein(a, b)


def _sample_exact(n, generator):
    """n real samples with exactly 20% in the small mode (for the measurements)."""
    k = torch.cat([torch.zeros(n // 5, dtype=torch.long), torch.ones(n - n // 5, dtype=torch.long)])
    return MUS[k] + torch.randn(n, 2, generator=generator)


# Fixed real samples for all measurements, with exactly 20% in the small mode, drawn with their own
# random generator so that they do not depend on what the notebook did before.
_g = torch.Generator().manual_seed(0)
X_EVAL = _sample_exact(1000, _g)
# What a perfect generator would get: the average distance between X_EVAL and 1000 fresh real samples.
_perfect = [sample_data(1000, generator=_g) for _ in range(10)]
W_FLOOR = float(np.mean([wasserstein(X_EVAL, x) for x in _perfect]))
W_FLOOR_LARGE = float(np.mean([wasserstein_large(X_EVAL, x) for x in _perfect]))


def print_floors():
    print(f"Wasserstein distance between X_EVAL and fresh real samples: {W_FLOOR:.2f}")
    print("(a perfect generator would get about this value, not 0, because of the finite samples)")
    print(f"Same, within the large mode only: {W_FLOOR_LARGE:.2f}")
    print(f"Fraction of the real samples in the small mode: {fraction_small_mode(X_EVAL):.0%}")


def new_history():
    return {"step": [], "small": [], "W": [], "W_large": [], "estimate": [], "loss_G": [], "loss_D": [], "samples": {}}


@torch.no_grad()
def record(history, step, G, D, losses, z_dim, n_steps):
    """Measures the current generator on 1000 samples and stores the results in history."""
    x_gen = G(torch.randn(len(X_EVAL), z_dim))
    history["step"].append(step)
    history["small"].append(fraction_small_mode(x_gen))
    history["W"].append(wasserstein(X_EVAL, x_gen))
    history["W_large"].append(wasserstein_large(X_EVAL, x_gen))
    history["estimate"].append((D(X_EVAL).mean() - D(x_gen).mean()).item())
    loss_D, loss_G = losses(D, X_EVAL, x_gen)
    history["loss_G"].append(loss_G.item())
    history["loss_D"].append(loss_D.item())
    if step in (n_steps // 6, n_steps // 3, 2 * n_steps // 3, n_steps):
        history["samples"][step] = x_gen.clone()


# ---------------------------------------------------------------- plots

COLORS = {"standard GAN": "tab:blue", "WGAN": "tab:green"}


def plot_snapshots(history, name):
    """The generated samples (red) over the real ones (blue), at 4 moments of the training."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (step, xs) in zip(axes, history["samples"].items()):
        plot_real(X_EVAL, ax)
        ax.scatter(xs[:, 0], xs[:, 1], s=4, alpha=0.5, color="#EE6677")
        ax.set_title(f"{name}, step {step}: {fraction_small_mode(xs):.0%} in the small mode")
    plt.show()


# Discriminator loss of the optimal discriminator against a generator that produces exactly the large mode:
# D* = 0.8/1.8 there and D* = 1 on the small mode, so L_D = -0.8 log(0.8/1.8) - log(1 - 0.8/1.8).
LOSS_D_COLLAPSED = -0.8 * math.log(0.8 / 1.8) - math.log(1 - 0.8 / 1.8)


def plot_curves(histories, third="loss_G"):
    """third="loss_G": generator and discriminator losses; third="gap": critic gap of the WGAN."""
    n = 4 if third == "loss_G" else 3
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    for name, h in histories.items():
        c = COLORS.get(name)
        axes[0].plot(h["step"], h["small"], marker="o", label=name, color=c)
        axes[1].plot(h["step"], h["W"], marker="o", label=f"{name}: all data", color=c)
        axes[1].plot(h["step"], h["W_large"], marker="s", linestyle=":", label=f"{name}: large mode only", color=c)
        if third == "loss_G":
            axes[2].plot(h["step"], h["loss_G"], marker="o", label=name, color=c)
            axes[3].plot(h["step"], h["loss_D"], marker="o", label=name, color=c)
        elif name == "WGAN":
            axes[2].plot(h["step"], h["estimate"], marker="o", label=name, color=c)
    axes[0].axhline(0.2, color="gray", linestyle="--", label="target: 20%")
    axes[0].set_title("fraction in the small mode")
    axes[0].set_ylim(0, 0.5)
    axes[1].axhline(W_FLOOR, color="gray", linestyle="--", label="perfect generator: all data")
    axes[1].axhline(W_FLOOR_LARGE, color="gray", linestyle=":", label="perfect generator: large mode only")
    axes[1].set_title("true Wasserstein distance to the data")
    if third == "loss_G":
        axes[2].axhline(-math.log(2), color="gray", linestyle="--", label="perfect generator, optimal D")
        lo = min(min(h["loss_G"]) for h in histories.values())
        hi = max(max(h["loss_G"]) for h in histories.values())
        axes[2].set_ylim(min(-1.0, lo - 0.05), max(-0.4, hi + 0.05))   # a fixed scale, so that small oscillations look small
        axes[3].axhline(LOSS_D_COLLAPSED, color="gray", linestyle="--", label="optimal D against N((5,5), I)")
        lo = min(min(h["loss_D"]) for h in histories.values())
        hi = max(max(h["loss_D"]) for h in histories.values())
        axes[3].set_ylim(min(1.0, lo - 0.05), max(1.6, hi + 0.05))
        axes[3].set_title("discriminator loss")
    axes[2].set_title("generator loss" if third == "loss_G" else "critic gap (WGAN)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend()
    plt.show()


def _hist_1d(ax):
    ax.hist(real_1d(5000).squeeze(), bins=60, density=True, alpha=0.3, color="#4477AA", label="real")
    ax.hist(fake_1d(5000).squeeze(), bins=60, density=True, alpha=0.3, color="#EE6677", label="fake")


def plot_1d(D_1d, f_1d):
    """The discriminator D = sigmoid(D_1d) and the critic f_1d, with their derivatives."""
    t = torch.linspace(-4, 4, 401).reshape(-1, 1).requires_grad_(True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (name, values) in zip(axes, [("discriminator $D$", torch.sigmoid(D_1d(t))),
                                         ("WGAN critic $f$ (1-Lipschitz)", f_1d(t))]):
        slope, = torch.autograd.grad(values.sum(), t)
        ax.plot(t.detach(), values.detach(), label="value")
        ax.plot(t.detach(), slope, label="derivative", linestyle="--")
        _hist_1d(ax)
        ax.set_title(name)
        ax.legend()
    plt.show()


def plot_1d_critic(f, label):
    t = torch.linspace(-4, 4, 401).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    with torch.no_grad():
        ax.plot(t, f(t), label=label)
    _hist_1d(ax)
    ax.legend()
    plt.show()


def compare_lipschitz(D_gan, f_sn, max_grad_norm, SNLinear):
    """Prints, for the 2-D discriminator and the 1-D spectrally normalized critic, the largest gradient
    norm on a grid (with your max_grad_norm) and the product of the spectral norms of the layers."""
    g = torch.linspace(-LIM, LIM, 91)
    grid_2d = torch.stack(torch.meshgrid(g, g, indexing="xy"), dim=-1).reshape(-1, 2)
    grid_1d = torch.linspace(-6, 6, 601).reshape(-1, 1)

    def spectral_bound(f):
        bound = 1.0
        for layer in f:
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach()
                if isinstance(layer, SNLinear):
                    W = W / torch.linalg.svdvals(W).max()
                bound *= torch.linalg.svdvals(W).max().item()
        return bound

    D_gan.eval()
    f_sn.eval()
    for name, f, points in [("standard GAN, raw score (2-D)", D_gan, grid_2d),
                            ("critic with SNLinear (1-D)", f_sn, grid_1d)]:
        print(f"{name:33s}  max |grad| = {max_grad_norm(f, points):7.3f}   "
              f"product of spectral norms = {spectral_bound(f):8.3f}")


# ---------------------------------------------------------------- checks

def _layer_types(net):
    return [type(m).__name__ for m in net]


def check_networks(build_generator, build_critic, z_dim, hidden):
    G = build_generator()
    assert G(torch.randn(7, z_dim)).shape == (7, 2), "the generator should map (n, Z_DIM) to (n, 2)"
    assert _layer_types(G) == ["Linear", "ReLU", "Linear", "ReLU", "Linear"], \
        f"generator layers: expected Linear, ReLU, Linear, ReLU, Linear (no final activation), got {_layer_types(G)}"
    assert G[2].in_features == hidden and G[2].out_features == hidden, "the hidden layers should have HIDDEN units"
    for d_in in [1, 2]:
        D = build_critic(d_in=d_in)
        assert D(torch.randn(7, d_in)).shape == (7, 1), f"the critic should map (n, {d_in}) to (n, 1)"
        assert _layer_types(D) == ["Linear", "ReLU", "Linear", "ReLU", "Linear"], \
            f"critic layers: expected Linear, ReLU, Linear, ReLU, Linear (no final activation), got {_layer_types(D)}"

    class MyLinear(nn.Linear):                          # another type of linear layer, to test the `linear` argument
        pass

    n_my = sum(isinstance(m, MyLinear) for m in build_critic(linear=MyLinear))
    assert n_my == 3, f"all 3 linear layers of the critic should be built with `linear`, found {n_my}"
    print("build_generator, build_critic: OK")


def check_gan_losses(gan_losses):
    # A discriminator that outputs 0 (i.e. D = 1/2 everywhere) gives loss_D = 2 log 2 and loss_G = log(1/2) = -log 2.
    zero_D = lambda x: torch.zeros(len(x), 1)
    loss_D, loss_G = gan_losses(zero_D, torch.randn(10, 2), torch.randn(10, 2))
    assert loss_D.ndim == 0 and loss_G.ndim == 0, "the losses should be single numbers (use .mean())"
    assert abs(loss_D.item() - 2 * math.log(2)) < 1e-5, f"loss_D should be 2 log 2 = 1.386 here, got {loss_D.item():.3f}"
    assert abs(loss_G.item() + math.log(2)) < 1e-5, \
        f"loss_G should be log(1 - 1/2) = -0.693 here (minimax loss), got {loss_G.item():.3f}"
    print("gan_losses: OK")


def check_wgan_losses(wgan_losses):
    lin = lambda x: x[:, :1]                            # f(x) = first coordinate
    loss_f, loss_G = wgan_losses(lin, torch.full((5, 2), 3.0), torch.full((5, 2), 1.0))
    assert loss_f.ndim == 0 and loss_G.ndim == 0, "the losses should be single numbers (use .mean())"
    assert abs(loss_f.item() + 2.0) < 1e-6, f"loss_f should be -(3 - 1) = -2 here, got {loss_f.item():.3f}"
    assert abs(loss_G.item() + 1.0) < 1e-6, f"loss_G should be -1 here, got {loss_G.item():.3f}"
    print("wgan_losses: OK")


def check_power_iteration(power_iteration):
    W = torch.randn(15, 15, requires_grad=True)
    u, v = power_iteration(W, torch.randn(15), 50)
    assert u.shape == (15,) and v.shape == (15,), "power_iteration should return u and v, two vectors"
    assert abs(u.norm().item() - 1) < 1e-4 and abs(v.norm().item() - 1) < 1e-4, "u and v should have norm 1"
    assert not u.requires_grad and not v.requires_grad, "do the updates under torch.no_grad()"
    exact = torch.linalg.svdvals(W.detach()).max().item()
    estimate = (u @ W.detach() @ v).item()
    assert abs(estimate - exact) < 1e-2 * exact, f"u^T W v should be about {exact:.3f} after 50 iterations, got {estimate:.3f}"
    print("power_iteration: OK")


def check_SNLinear(SNLinear):
    # After a few passes, the normalized weight should have largest singular value 1.
    layer = SNLinear(10, 20)
    with torch.no_grad():
        layer.weight.mul_(5.0)                          # make sigma(W) large
    x = torch.randn(7, 10)
    for _ in range(100):
        out = layer(x)
    assert out.shape == (7, 20), f"expected output shape (7, 20), got {tuple(out.shape)}"
    W = layer.weight.detach()
    sigma_exact = torch.linalg.svdvals(W).max()
    sigma_layer = layer.u @ W @ F.normalize(W.t() @ layer.u, dim=0)
    assert abs(sigma_layer.item() / sigma_exact.item() - 1) < 1e-2, "the estimate of sigma(W) has not converged"
    assert torch.allclose(out, x @ (W / sigma_exact).t() + layer.bias, atol=1e-2), \
        "the layer should compute x W^T / sigma(W) + b"
    print("SNLinear: OK")


def check_max_grad_norm(max_grad_norm):
    lin = lambda x: (3 * x[:, 0] - 4 * x[:, 1]).unsqueeze(1)    # gradient (3, -4), of norm 5
    assert abs(max_grad_norm(lin, torch.randn(50, 2)) - 5.0) < 1e-5, "expected 5"
    print("max_grad_norm: OK")


# ---------------------------------------------------------------- Section 4: oscillations

def plot_toy_gan(runs, lim=None, equilibrium=0.0):
    """runs: {name: (thetas, psis)}. Left: theta over time. Right: the trajectory in the (theta, psi) plane.
    lim: if given, both panels are limited to [-lim, lim] (diverging runs then leave the frame)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for name, (thetas, psis) in runs.items():
        line, = axes[0].plot(thetas, label=name)
        axes[1].plot(thetas, psis, label=name, color=line.get_color(), linewidth=1)
        axes[1].plot(thetas[0], psis[0], "o", color="black")
    axes[0].axhline(equilibrium, color="gray", linestyle="--", label="equilibrium")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel(r"generator $\theta$")
    axes[0].set_title(r"position of the generator, $\theta$")
    axes[1].plot(equilibrium, 0, "*", color="black", markersize=12, label="equilibrium")
    axes[1].set_xlabel(r"generator $\theta$")
    axes[1].set_ylabel(r"critic slope $\psi$")
    axes[1].set_title(r"trajectory in the $(\theta, \psi)$ plane (black dot: start)")
    if lim is None:
        axes[1].set_aspect("equal", adjustable="datalim")
    else:
        axes[0].set_ylim(-lim, lim)
        axes[1].set_xlim(-lim, lim)
        axes[1].set_ylim(-lim, lim)
        axes[1].set_aspect("equal")
    for ax in axes:
        ax.legend()
    plt.show()


def check_train_toy_gan(train_toy_gan):
    rng = np.random.default_rng(1)
    x, z = rng.standard_normal(50), rng.standard_normal(50)
    gap = z.mean() - x.mean()

    def reference(h, n_steps, n_critic):
        theta, psi = 1.0, 0.0
        for _ in range(n_steps):
            for _ in range(n_critic):
                psi = max(-1.0, min(1.0, psi - h * (theta + gap)))
            theta = theta + h * psi
        return theta, psi
    for kwargs in [dict(n_steps=1, n_critic=1), dict(n_steps=3, n_critic=2), dict(n_steps=5, n_critic=5)]:
        thetas, psis = train_toy_gan(x, z, h=0.1, **kwargs)
        assert len(thetas) == kwargs["n_steps"] + 1 and len(psis) == kwargs["n_steps"] + 1, \
            "return the values of theta and psi at the start and after each step"
        t, p = reference(0.1, **kwargs)
        assert abs(float(thetas[-1]) - t) < 1e-6 and abs(float(psis[-1]) - p) < 1e-6, \
            f"with {kwargs}: expected theta = {t:.4f}, psi = {p:.4f} at the end, got {float(thetas[-1]):.4f}, {float(psis[-1]):.4f}" \
            " (the critic moves first, n_critic times, with psi clipped to [-1, 1]; then the generator uses the new psi)"
    print("train_toy_gan: OK")
