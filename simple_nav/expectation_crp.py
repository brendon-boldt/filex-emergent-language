from typing import Callable

import numpy as np
from matplotlib import pyplot as plt  # type: ignore
import matplotlib
from tqdm import tqdm  # type: ignore
from joblib import Parallel, delayed  # type: ignore
import argparse
from scipy.stats import kendalltau  # type: ignore

from .experiment_configs import log_range


def entropy(counts):
    x = counts / max(counts.sum(), 1)
    x = x[x > 0]
    return -(x * np.log2(x)).sum()


def do_run(
    alpha,
    beta,
    time_steps,
):
    counts = np.zeros((max_size,), dtype=np.float64)
    counts[0] = 1.0
    xs = []
    rng = np.random.default_rng()
    for i in range(time_steps):
        probs = counts.copy()
        probs[np.flatnonzero(probs == 0)[0]] = alpha
        probs /= probs.sum()
        counts += eye[rng.choice(len(counts), beta, p=probs)].sum(0) / beta
    return entropy(counts)


def main(args) -> None:
    xs = []
    ys = []
    jobs = []

    if args.data_path is None:
        global max_size, n, eye
        max_size = 2**12
        eye = np.eye(max_size)
        n = 10_000
        params = {
            "alpha": 5.0,
            "beta": int(1e2),
            "time_steps": int(1e3),
        }
        lo = 1
        hi = 1000
        for val in log_range(lo, hi, n):
            xs.append(val)
            val = int(val)
            params["beta"] = val
            jobs.append(delayed(do_run)(**params))

        ys = Parallel(n_jobs=args.j)(j for j in tqdm(jobs))
        alpha = 0.02
    else:
        data = np.genfromtxt(args.data_path, delimiter=",")
        xs, ys = data.transpose()
        # ys = np.array(2.0) ** ys
        alpha = min(1, 0.02 * 10_000 / len(xs))

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    # ax2 = ax.twinx()

    ax.set_ylim(-0.5, 6.5)

    ax.set_ylabel("Entropy (bits)")
    ax.scatter(np.log10(xs), ys, s=2.0, alpha=alpha, color="Orange")

    func: Callable
    if args.config_name == "alpha":
        ticks = [0.01, 1, 100]
    elif args.config_name == "beta":
        ticks = [1, 10, 100, 1000]
    elif args.config_name == "n_iters":
        ticks = [1, 10, 100, 1000]
    elif args.config_name == "n_params":
        ticks = [0x8, 0x20, 0x80]
    else:
        raise ValueError(f"Config name {args.config_name} not found")

    ax.set_xticks([np.log10(x) for x in ticks])
    ax.set_xticklabels(ticks)

    fig.savefig(
        f"results/model-{args.config_name}.png",
        bbox_inches="tight",
        format="png",
        dpi=600,
    )
    fig.savefig(
        f"results/model-{args.config_name}.pdf",
        bbox_inches="tight",
        format="pdf",
        dpi=600,
    )

    result = kendalltau(xs, ys)
    print(f"correlation: {result.correlation:.2f}\t" f"p-value: {result.pvalue:.2f}\t")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("-j", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
