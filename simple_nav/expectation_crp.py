import numpy as np
from matplotlib import pyplot as plt  # type: ignore
import argparse
from scipy.stats import kendalltau  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore

from .experiment_configs import log_range


def entropy(counts):
    x = counts / max(counts.sum(), 1)
    x = x[x > 0]
    return -(x * np.log2(x)).sum()


def main(args) -> None:
    data = np.genfromtxt(args.data_path, delimiter=",")
    xs, ys = data.transpose()
    alpha = min(1, 0.02 * 10_000 / len(xs))

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_ylim(-0.5, 7.0)
    # ax.set_ylabel("Entropy (bits)")
    lxs = np.log10(xs)

    if args.config_name != "n_params":
        ax.plot([lxs.min(), lxs.max()], [6, 6], color="Gray", alpha=0.2, linestyle='--')
        ax.plot([lxs.min(), lxs.max()], [0, 0], color="Gray", alpha=0.2, linestyle='--')

    ax.scatter(lxs, ys, s=2.0, edgecolors= None, alpha=alpha, color="Gray")
    ax.plot(lxs, gaussian_filter(ys, sigma=30), color="Orange")

    result = kendalltau(xs, ys)
    print(f"correlation: {result.correlation:+.2f}\t" f"p-value: {result.pvalue:.2f}\t")

    sgn = "−" if result.correlation < 0 else "+"
    val = abs(result.correlation)
    ax.text(
        0.05,
        0.9,
        f"τ: {sgn}{val:.2f}",
        transform=ax.transAxes,
        fontfamily="monospace",
    )


    if args.config_name == "alpha":
        ticks = [0.01, 1, 100]
    elif args.config_name == "beta":
        ticks = [1, 10, 100, 1000]
    elif args.config_name == "n_iters":
        ticks = [1, 10, 100, 1000]
    elif args.config_name == "n_params":
        ticks = [0x8, 0x20, 0x80]
    elif args.config_name == "scratch":
        pass
    else:
        raise ValueError(f"Config name {args.config_name} not found")

    ax.set_xticks([])
    ax.set_yticks([])

    # ax.set_xticks([np.log10(x) for x in ticks])
    # ax.set_xticklabels(ticks)

    for fmt in ["pdf", "png"]:
        fig.savefig(
            f"results/model-{args.config_name}.{fmt}",
            bbox_inches="tight",
            format=fmt,
            dpi=600,
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("-j", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
