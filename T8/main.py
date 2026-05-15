import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniform U[0, theta] simulation: optimal estimator vs method of moments."
    )
    parser.add_argument("--n", type=int, default=1000, help="Sample size.")
    parser.add_argument("--reps", type=int, default=101, help="Repetitions per run.")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs.")
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


def simulate_stats(rng: np.random.Generator, theta: float, n: int, reps: int):
    optimal = np.empty(reps)
    moments = np.empty(reps)
    for i in range(reps):
        sample = rng.uniform(0.0, theta, size=n)
        optimal[i] = (n + 1) / n * sample.max()
        moments[i] = 2.0 * sample.mean()
    return optimal, moments


def plot_hist(plt, data: np.ndarray, theta: float, title: str, xlabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=20, color="#4C78A8", edgecolor="black", alpha=0.8)
    ax.axvline(theta, color="red", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_single_experiment(
    plt, rng: np.random.Generator, run_index: int, n: int, reps: int, outdir: Path
):
    theta = rng.uniform(1.0, 10.0)
    optimal, moments = simulate_stats(rng, theta, n, reps)

    plot_hist(
        plt,
        optimal,
        theta,
        f"Run {run_index}: optimal estimator (theta={theta:.3f})",
        "(n+1)/n * X_(n)",
        outdir / f"run_{run_index}_optimal.png",
    )
    plot_hist(
        plt,
        moments,
        theta,
        f"Run {run_index}: method of moments (theta={theta:.3f})",
        "2 * X_bar",
        outdir / f"run_{run_index}_moments.png",
    )

    scaled_optimal = n * (optimal - theta)
    scaled_moments = np.sqrt(n) * (moments - theta)

    plot_hist(
        plt,
        scaled_optimal,
        0.0,
        f"Run {run_index}: n*(theta_hat_1 - theta)",
        "n*(theta_hat_1 - theta)",
        outdir / f"run_{run_index}_optimal_scaled.png",
    )
    plot_hist(
        plt,
        scaled_moments,
        0.0,
        f"Run {run_index}: sqrt(n)*(theta_hat_2 - theta)",
        "sqrt(n)*(theta_hat_2 - theta)",
        outdir / f"run_{run_index}_moments_scaled.png",
    )

    th_var_opt = theta ** 2 / (n * (n + 2))
    th_var_mom = theta ** 2 / (3 * n)

    print(f"Run {run_index}: theta = {theta:.6f}")
    print(
        "  Optimal:  mean={:.6f}, std={:.6f}  (theory SD = {:.6f})".format(
            optimal.mean(), optimal.std(ddof=1), np.sqrt(th_var_opt)
        )
    )
    print(
        "  Moments:  mean={:.6f}, std={:.6f}  (theory SD = {:.6f})".format(
            moments.mean(), moments.std(ddof=1), np.sqrt(th_var_mom)
        )
    )
    print("  Conclusion: optimal converges as 1/n^2 (faster than 1/n of MoM).\n")


def main() -> None:
    args = parse_args()
    plt = configure_matplotlib(args.show)

    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for run_index in range(1, args.runs + 1):
        run_single_experiment(plt, rng, run_index, args.n, args.reps, args.outdir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
