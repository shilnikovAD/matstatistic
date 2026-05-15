import argparse
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kolmogorov-Smirnov statistic: D_n trajectory and limit distribution."
    )
    parser.add_argument("--n-max", type=int, default=1000, help="Maximum sample size for trajectory.")
    parser.add_argument("--mc-reps", type=int, default=2000,
                        help="Monte Carlo replications for limit distribution.")
    parser.add_argument("--mc-n", type=int, default=1000,
                        help="Sample size used in Monte Carlo block.")
    parser.add_argument("--x-min", type=float, default=-1.0, help="Grid lower bound for CDF plot.")
    parser.add_argument("--x-max", type=float, default=10.0, help="Grid upper bound for CDF plot.")
    parser.add_argument("--x-points", type=int, default=1100, help="Number of grid points for CDF plot.")
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


def ks_stat_from_sorted(sorted_sample: np.ndarray) -> float:
    """KS statistic D_n for a sorted sample assuming F = standard normal CDF."""
    n = sorted_sample.size
    F_vals = stats.norm.cdf(sorted_sample)
    i = np.arange(1, n + 1)
    d_plus = (i / n - F_vals).max()
    d_minus = (F_vals - (i - 1) / n).max()
    return float(max(d_plus, d_minus))


def trajectory_D(sample: np.ndarray) -> np.ndarray:
    """For each n=1..len(sample) returns D_n based on the first n observations."""
    n_max = sample.size
    D = np.empty(n_max)
    for n in range(1, n_max + 1):
        D[n - 1] = ks_stat_from_sorted(np.sort(sample[:n]))
    return D


def monte_carlo_sqrtnD(rng: np.random.Generator, n: int, reps: int) -> np.ndarray:
    out = np.empty(reps)
    for r in range(reps):
        sample = rng.standard_normal(size=n)
        sample.sort()
        out[r] = np.sqrt(n) * ks_stat_from_sorted(sample)
    return out


def plot_trajectory(plt, ns: np.ndarray, D: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, D, color="#4C78A8", linewidth=0.9, label="D_n")
    ref = 1.0 / np.sqrt(ns)
    ax.plot(ns, ref, color="#999999", linestyle="--", linewidth=1.0, label="1 / sqrt(n)")
    ax.set_title("KS statistic D_n vs n  (single sample, F = N(0,1))")
    ax.set_xlabel("n")
    ax.set_ylabel("D_n")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cdf_comparison(plt, xs: np.ndarray, ecdf: np.ndarray, K_vals: np.ndarray,
                        n: int, reps: int, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ecdf, color="#4C78A8", linewidth=1.5,
            label=f"empirical CDF of sqrt(n)*D_n  (n={n}, {reps} reps)")
    ax.plot(xs, K_vals, color="red", linestyle="--", linewidth=1.5,
            label="Kolmogorov CDF K(x)")
    ax.set_title("Empirical CDF of sqrt(n)*D_n vs Kolmogorov limit")
    ax.set_xlabel("x")
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt = configure_matplotlib(args.show)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Part 1-2: single sample, plot D_n vs n
    sample = rng.standard_normal(size=args.n_max)
    D = trajectory_D(sample)
    ns = np.arange(1, args.n_max + 1)
    plot_trajectory(plt, ns, D, args.outdir / "Dn_trajectory.png")

    # Part 3: Monte Carlo for sqrt(n)*D_n vs Kolmogorov K(x)
    sqrtnD = monte_carlo_sqrtnD(rng, args.mc_n, args.mc_reps)
    xs = np.linspace(args.x_min, args.x_max, args.x_points)
    sorted_stat = np.sort(sqrtnD)
    ecdf = np.searchsorted(sorted_stat, xs, side="right") / sqrtnD.size
    K_vals = stats.kstwobign.cdf(xs)
    plot_cdf_comparison(plt, xs, ecdf, K_vals, args.mc_n, args.mc_reps,
                        args.outdir / "sqrtnDn_vs_kolmogorov.png")

    sup_diff = float(np.max(np.abs(ecdf - K_vals)))
    print(f"Trajectory: n in [1, {args.n_max}]")
    print(f"  D_1   = {D[0]:.6f}")
    print(f"  D_10  = {D[9]:.6f}   (1/sqrt(10)  = {1.0/np.sqrt(10):.6f})")
    print(f"  D_100 = {D[99]:.6f}  (1/sqrt(100) = {1.0/np.sqrt(100):.6f})")
    print(f"  D_{args.n_max} = {D[-1]:.6f}  (1/sqrt({args.n_max}) = {1.0/np.sqrt(args.n_max):.6f})")
    print(f"  max D_n over n >= 200: {np.max(D[199:]):.6f}")
    print()
    print(f"Monte Carlo: n = {args.mc_n}, reps = {args.mc_reps}")
    print(f"  mean(sqrt(n)*D_n) = {sqrtnD.mean():.5f}  (Kolmogorov mean ~ sqrt(pi/2)*ln(2) ~ 0.8687)")
    print(f"  std(sqrt(n)*D_n)  = {sqrtnD.std(ddof=1):.5f}")
    print(f"  sup |ecdf - K|    = {sup_diff:.5f}   (uniform error vs Kolmogorov limit)")
    for q in (0.5, 0.75, 0.90, 0.95, 0.99):
        emp = float(np.quantile(sqrtnD, q))
        the = float(stats.kstwobign.ppf(q))
        print(f"  quantile {q:.2f}: empirical {emp:.5f}   theoretical {the:.5f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
