import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normal location simulation: median vs mean."
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
    medians = np.empty(reps)
    means = np.empty(reps)
    for i in range(reps):
        sample = rng.normal(loc=theta, scale=1.0, size=n)
        medians[i] = np.median(sample)
        means[i] = np.mean(sample)
    return medians, means


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
    theta = rng.uniform(-10.0, 10.0)
    medians, means = simulate_stats(rng, theta, n, reps)

    plot_hist(
        plt,
        medians,
        theta,
        f"Run {run_index}: sample median (theta={theta:.3f})",
        "Median",
        outdir / f"run_{run_index}_median.png",
    )
    plot_hist(
        plt,
        means,
        theta,
        f"Run {run_index}: sample mean (theta={theta:.3f})",
        "Mean",
        outdir / f"run_{run_index}_mean.png",
    )

    scaled_medians = np.sqrt(n) * (medians - theta)
    scaled_means = np.sqrt(n) * (means - theta)

    plot_hist(
        plt,
        scaled_medians,
        0.0,
        f"Run {run_index}: sqrt(n)*(median-theta)",
        "sqrt(n)*(median-theta)",
        outdir / f"run_{run_index}_median_scaled.png",
    )
    plot_hist(
        plt,
        scaled_means,
        0.0,
        f"Run {run_index}: sqrt(n)*(mean-theta)",
        "sqrt(n)*(mean-theta)",
        outdir / f"run_{run_index}_mean_scaled.png",
    )

    print(f"Run {run_index}: theta = {theta:.6f}")
    print(
        "  Median: mean={:.6f}, std={:.6f}".format(
            medians.mean(), medians.std(ddof=1)
        )
    )
    print(
        "  Mean:   mean={:.6f}, std={:.6f}".format(
            means.mean(), means.std(ddof=1)
        )
    )
    print("  Conclusion: mean is more efficient for Normal.\n")


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

