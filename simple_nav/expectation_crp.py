import numpy as np
from matplotlib import pyplot as plt  # type: ignore
import matplotlib
from tqdm import tqdm  # type: ignore
from joblib import Parallel, delayed  # type: ignore
import argparse

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
        max_size = 2 ** 12
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
    ax2 = ax.twinx()

    # window size
    ws = 100  # on either side
    stddevs = [
            # ys[i-ws:i+ws].std()
            (((ys[i-ws:i+ws-1] - ys[i-ws+1:i+ws] ) ** 2).sum() / (2*ws)) ** 0.5
            for i in range(ws, len(ys) - ws)
            ]
    ax.plot(np.log10(xs[ws:-ws]), stddevs, color='Orange', alpha=0.5)

    ax2.scatter(np.log10(xs), ys, s=2.0, alpha=alpha)

    # ticks = [1, 10, 100, 1000]
    # ax.set_xticks([np.log10(x) for x in ticks])
    # ax.set_xticklabels(ticks)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Entropy (bits)")

    fig.savefig("results/ecrp.png", bbox_inches="tight", format="png", dpi=600)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, nargs="?")
    parser.add_argument("-j", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
