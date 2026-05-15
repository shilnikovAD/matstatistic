import argparse
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact confidence intervals for the mean of N(theta, 1) across a theta grid."
    )
    parser.add_argument(
        "--ns", type=int, nargs="+", default=[5, 10, 100, 200],
        help="Sample sizes to use (one plot per n).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Known noise standard deviation.")
    parser.add_argument("--theta-min", type=float, default=-10.0, help="Lower bound for theta grid.")
    parser.add_argument("--theta-max", type=float, default=10.0, help="Upper bound for theta grid.")
    parser.add_argument("--theta-step", type=float, default=0.1, help="Step of theta grid.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument(
        "--outdir", type=Path, default=Path("out"), help="Output directory for PNGs."
    )
    parser.add_argument(
        "--show", action="store_true", help="Show figures interactively."
    )
    return parser.parse_args()


def configure_matplotlib(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_theta_grid(theta_min: float, theta_max: float, step: float) -> np.ndarray:
    k_lo = int(round(theta_min / step))
    k_hi = int(round(theta_max / step))
    k = np.arange(k_lo, k_hi + 1)
    return k * step


def confidence_intervals(rng: np.random.Generator, thetas: np.ndarray,
                         n: int, sigma: float, alpha: float):
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    half = z * sigma / np.sqrt(n)
    x_bar = np.empty(thetas.size)
    for i, theta in enumerate(thetas):
        sample = rng.normal(theta, sigma, size=n)
        x_bar[i] = sample.mean()
    return x_bar - half, x_bar + half, x_bar, half, z


def plot_intervals(plt, thetas: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   x_bar: np.ndarray, n: int, alpha: float, half: float,
                   sigma: float, coverage: float, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    inside = (lower <= thetas) & (thetas <= upper)
    ax.vlines(thetas[inside], lower[inside], upper[inside],
              color="#4C78A8", alpha=0.7, linewidth=0.9, label="CI (covers theta)")
    if (~inside).any():
        ax.vlines(thetas[~inside], lower[~inside], upper[~inside],
                  color="#E45756", alpha=0.9, linewidth=1.1, label="CI (misses theta)")
    ax.plot(thetas, x_bar, "o", color="#4C78A8", markersize=1.6, alpha=0.7)
    lim = (thetas.min() - 0.5, thetas.max() + 0.5)
    ax.plot(lim, lim, color="red", linestyle="--", linewidth=1.2,
            label="f(theta) = theta")
    ax.set_title(
        f"n = {n}, sigma = {sigma}, alpha = {alpha}: "
        f"half-width = {half:.4f}, length = {2*half:.4f}, "
        f"coverage = {coverage:.3f}"
    )
    ax.set_xlabel("theta")
    ax.set_ylabel("CI for theta")
    ax.set_xlim(lim)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_length_vs_n(plt, ns: np.ndarray, lengths: np.ndarray,
                     z: float, sigma: float, path: Path):
    grid = np.linspace(ns.min(), ns.max(), 200)
    theoretical = 2.0 * z * sigma / np.sqrt(grid)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, theoretical, color="red", linestyle="--",
            label=f"2*z*sigma/sqrt(n), z = {z:.4f}")
    ax.plot(ns, lengths, "o", color="#4C78A8", markersize=8, label="empirical (= 2*half)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("CI length vs n  (log-log)")
    ax.set_xlabel("n")
    ax.set_ylabel("CI length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt = configure_matplotlib(args.show)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    thetas = build_theta_grid(args.theta_min, args.theta_max, args.theta_step)
    z = stats.norm.ppf(1.0 - args.alpha / 2.0)

    print(f"alpha = {args.alpha},  z_(1-alpha/2) = {z:.6f}")
    print(f"theta grid: {thetas[0]:.2f} ... {thetas[-1]:.2f} "
          f"(step {args.theta_step}, {len(thetas)} points)")
    print()

    lengths = []
    for n in args.ns:
        lower, upper, x_bar, half, _ = confidence_intervals(
            rng, thetas, n, args.sigma, args.alpha
        )
        coverage = float(np.mean((lower <= thetas) & (thetas <= upper)))
        plot_intervals(
            plt, thetas, lower, upper, x_bar, n, args.alpha, half,
            args.sigma, coverage, args.outdir / f"ci_n{n}.png",
        )
        lengths.append(2.0 * half)
        print(
            f"n = {n:4d}: half-width = {half:.6f}  "
            f"(z*sigma/sqrt(n) = {z*args.sigma/np.sqrt(n):.6f}),  "
            f"length = {2*half:.6f},  coverage = {coverage:.4f}"
        )

    plot_length_vs_n(
        plt, np.array(args.ns, dtype=float), np.array(lengths),
        z, args.sigma, args.outdir / "length_vs_n.png",
    )

    print()
    print("Analytical: length = 2 * z_(1-alpha/2) * sigma / sqrt(n) ~ 1/sqrt(n).")
    print("Empirical coverage should fluctuate around 1 - alpha = "
          f"{1 - args.alpha:.2f}.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
