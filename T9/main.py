import argparse
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-weight Gaussian linear model: F-tests and optimal estimators."
    )
    parser.add_argument("--n-max", type=int, default=200, help="Maximum sample size.")
    parser.add_argument("--theta1", type=float, default=5.0, help="True theta1.")
    parser.add_argument("--theta2", type=float, default=8.0, help="True theta2.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise standard deviation.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--mc-reps", type=int, default=2000,
        help="Monte Carlo replications for variance comparison.",
    )
    parser.add_argument(
        "--mc-n", type=int, default=200,
        help="Sample size used in the variance comparison block.",
    )
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


def simulate_one(rng: np.random.Generator, theta1: float, theta2: float,
                 sigma: float, n: int):
    X = rng.normal(theta1, sigma, size=n)
    Y = rng.normal(theta2, sigma, size=2 * n)
    Z = rng.normal(theta1 + theta2, sigma, size=n)
    return X, Y, Z


def fit_full(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, n: int):
    sum_x, sum_y, sum_z = X.sum(), Y.sum(), Z.sum()
    theta1_hat = (3.0 * sum_x - sum_y + 2.0 * sum_z) / (5.0 * n)
    theta2_hat = (-sum_x + 2.0 * sum_y + sum_z) / (5.0 * n)
    rss = (
        ((X - theta1_hat) ** 2).sum()
        + ((Y - theta2_hat) ** 2).sum()
        + ((Z - theta1_hat - theta2_hat) ** 2).sum()
    )
    return theta1_hat, theta2_hat, rss


def fit_constrained(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                    n: int, a: float, b: float):
    # H0: a*theta1 = b*theta2  =>  theta2 = (a/b) * theta1
    r = a / b
    s = 1.0 + r
    denom = n * (1.0 + 2.0 * r * r + s * s)
    theta1_r = (X.sum() + r * Y.sum() + s * Z.sum()) / denom
    theta2_r = r * theta1_r
    rss = (
        ((X - theta1_r) ** 2).sum()
        + ((Y - theta2_r) ** 2).sum()
        + ((Z - theta1_r - theta2_r) ** 2).sum()
    )
    return theta1_r, theta2_r, rss


def f_test_reject(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                  n: int, a: float, b: float, alpha: float) -> bool:
    _, _, rss_full = fit_full(X, Y, Z, n)
    _, _, rss_red = fit_constrained(X, Y, Z, n, a, b)
    q = 1
    df_resid = 4 * n - 2
    if df_resid <= 0 or rss_full <= 0.0:
        return False
    F = ((rss_red - rss_full) / q) / (rss_full / df_resid)
    p_value = 1.0 - stats.f.cdf(F, q, df_resid)
    return p_value < alpha


def plot_fraction(plt, ns: np.ndarray, fraction: np.ndarray, target: float,
                  target_label: str, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, fraction, color="#4C78A8", linewidth=1.2, label="empirical fraction")
    ax.axhline(target, color="red", linestyle="--", label=target_label)
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("fraction of non-rejections")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_estimators(plt, ns: np.ndarray, optimal: np.ndarray, naive: np.ndarray,
                    truth: float, naive_label: str, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, naive, color="#F58518", linewidth=1.0, alpha=0.85, label=naive_label)
    ax.plot(ns, optimal, color="#4C78A8", linewidth=1.0, alpha=0.95, label="optimal")
    ax.axhline(truth, color="red", linestyle="--", label=f"truth = {truth}")
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("estimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hist_pair(plt, optimal: np.ndarray, naive: np.ndarray, truth: float,
                   naive_label: str, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = 30
    ax.hist(naive, bins=bins, color="#F58518", alpha=0.5, edgecolor="black",
            label=naive_label)
    ax.hist(optimal, bins=bins, color="#4C78A8", alpha=0.6, edgecolor="black",
            label="optimal")
    ax.axvline(truth, color="red", linewidth=2, label=f"truth = {truth}")
    ax.set_title(title)
    ax.set_xlabel("estimate")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def monte_carlo_variance(rng: np.random.Generator, theta1: float, theta2: float,
                         sigma: float, n: int, reps: int):
    opt1 = np.empty(reps)
    opt2 = np.empty(reps)
    nai1 = np.empty(reps)
    nai2 = np.empty(reps)
    for i in range(reps):
        X, Y, Z = simulate_one(rng, theta1, theta2, sigma, n)
        t1, t2, _ = fit_full(X, Y, Z, n)
        opt1[i] = t1
        opt2[i] = t2
        nai1[i] = X.mean()
        nai2[i] = Y.mean()
    return opt1, opt2, nai1, nai2


def main() -> None:
    args = parse_args()
    plt = configure_matplotlib(args.show)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ns = np.arange(1, args.n_max + 1)
    xi_h1 = np.empty(len(ns), dtype=int)  # H0: 8*theta1 = 5*theta2 (TRUE)
    xi_h2 = np.empty(len(ns), dtype=int)  # H0: 2*theta1 = 3*theta2 (FALSE)
    theta1_opt = np.empty(len(ns))
    theta2_opt = np.empty(len(ns))
    x_bar = np.empty(len(ns))
    y_bar = np.empty(len(ns))

    for idx, n in enumerate(ns):
        n_int = int(n)
        X, Y, Z = simulate_one(rng, args.theta1, args.theta2, args.sigma, n_int)
        t1, t2, _ = fit_full(X, Y, Z, n_int)
        theta1_opt[idx] = t1
        theta2_opt[idx] = t2
        x_bar[idx] = X.mean()
        y_bar[idx] = Y.mean()
        xi_h1[idx] = 0 if f_test_reject(X, Y, Z, n_int, 8.0, 5.0, args.alpha) else 1
        xi_h2[idx] = 0 if f_test_reject(X, Y, Z, n_int, 2.0, 3.0, args.alpha) else 1

    frac_h1 = np.cumsum(xi_h1) / ns
    frac_h2 = np.cumsum(xi_h2) / ns

    plot_fraction(
        plt, ns, frac_h1, 1.0 - args.alpha,
        f"1 - alpha = {1.0 - args.alpha:.2f}",
        "H0: 8*theta1 = 5*theta2 (TRUE) — running fraction of non-rejections",
        args.outdir / "h0_8t1_eq_5t2_fraction.png",
    )
    plot_fraction(
        plt, ns, frac_h2, 0.0,
        "expected limit: 0 (power -> 1)",
        "H0: 2*theta1 = 3*theta2 (FALSE) — running fraction of non-rejections",
        args.outdir / "h0_2t1_eq_3t2_fraction.png",
    )

    plot_estimators(
        plt, ns, theta1_opt, x_bar, args.theta1,
        "X_bar = sum(X)/n",
        "theta1 estimators across n (single realization per n)",
        args.outdir / "theta1_path.png",
    )
    plot_estimators(
        plt, ns, theta2_opt, y_bar, args.theta2,
        "Y_bar = sum(Y)/(2n)",
        "theta2 estimators across n (single realization per n)",
        args.outdir / "theta2_path.png",
    )

    # Monte Carlo: variance comparison at fixed n
    opt1, opt2, nai1, nai2 = monte_carlo_variance(
        rng, args.theta1, args.theta2, args.sigma, args.mc_n, args.mc_reps
    )

    plot_hist_pair(
        plt, opt1, nai1, args.theta1, "X_bar = sum(X)/n",
        f"theta1 estimators at n={args.mc_n} ({args.mc_reps} MC reps)",
        args.outdir / "theta1_hist.png",
    )
    plot_hist_pair(
        plt, opt2, nai2, args.theta2, "Y_bar = sum(Y)/(2n)",
        f"theta2 estimators at n={args.mc_n} ({args.mc_reps} MC reps)",
        args.outdir / "theta2_hist.png",
    )

    sd_opt1_th = args.sigma * np.sqrt(3.0 / (5.0 * args.mc_n))
    sd_nai1_th = args.sigma * np.sqrt(1.0 / args.mc_n)
    sd_opt2_th = args.sigma * np.sqrt(2.0 / (5.0 * args.mc_n))
    sd_nai2_th = args.sigma * np.sqrt(1.0 / (2.0 * args.mc_n))

    final_n = int(ns[-1])
    print(f"theta1 = {args.theta1}, theta2 = {args.theta2}, sigma = {args.sigma}, "
          f"alpha = {args.alpha}, n in [1, {args.n_max}]")
    print()
    print("Part 1-2.  H0: 8*theta1 = 5*theta2  (TRUE: 8*5 = 5*8 = 40)")
    print(f"  Running fraction of non-rejections at n={final_n}: {frac_h1[-1]:.4f}")
    print(f"  Expected long-run value:                          {1.0 - args.alpha:.4f}")
    print()
    print("Part 3.    H0: 2*theta1 = 3*theta2  (FALSE: 2*5 = 10, 3*8 = 24)")
    print(f"  Running fraction of non-rejections at n={final_n}: {frac_h2[-1]:.4f}")
    print(f"  Expected long-run value:                          ~0 (power -> 1)")
    print()
    print(f"Part 4.    theta1 estimators (MC at n={args.mc_n}, {args.mc_reps} reps)")
    print(f"  Optimal:  mean={opt1.mean():.5f}, "
          f"std={opt1.std(ddof=1):.5f}  (theory SD = {sd_opt1_th:.5f})")
    print(f"  X_bar:    mean={nai1.mean():.5f}, "
          f"std={nai1.std(ddof=1):.5f}  (theory SD = {sd_nai1_th:.5f})")
    print(f"  Var(optimal)/Var(X_bar) (empirical):  "
          f"{(opt1.var(ddof=1) / nai1.var(ddof=1)):.4f}    (theory: 3/5 = 0.6)")
    print()
    print(f"Part 5.    theta2 estimators (MC at n={args.mc_n}, {args.mc_reps} reps)")
    print(f"  Optimal:  mean={opt2.mean():.5f}, "
          f"std={opt2.std(ddof=1):.5f}  (theory SD = {sd_opt2_th:.5f})")
    print(f"  Y_bar:    mean={nai2.mean():.5f}, "
          f"std={nai2.std(ddof=1):.5f}  (theory SD = {sd_nai2_th:.5f})")
    print(f"  Var(optimal)/Var(Y_bar) (empirical):  "
          f"{(opt2.var(ddof=1) / nai2.var(ddof=1)):.4f}    (theory: 4/5 = 0.8)")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()