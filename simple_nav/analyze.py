import argparse
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union, Iterator, Optional, Callable
from itertools import product
import math
from joblib import Parallel, delayed  # type: ignore

from scipy.stats import linregress, kendalltau  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib

from . import analysis_configs
from .util import log_range


def iter_groups(
    df: pd.DataFrame,
    groups: List[str],
    plot_shape: Optional[Tuple[int, int]],
    no_axes=False,
) -> Iterator[Tuple[List, pd.DataFrame, matplotlib.axes.Axes]]:
    valss = product(*(df[groups[i]].unique() for i in range(len(groups))))
    for vals in valss:
        filtered = df.loc[(df[groups] == vals).all(1)]
        if not len(filtered):
            continue
        if plot_shape is not None:
            random_idxs = True
            if random_idxs:
                random_idxs = np.random.default_rng().choice(
                    len(filtered),
                    min(len(filtered), np.prod(plot_shape)),
                    replace=False,
                )
                filtered = filtered.iloc[random_idxs]
            else:
                filtered = filtered[: plot_shape[0] * plot_shape[1]]
        else:
            row_len = math.ceil(math.sqrt(len(filtered)))
            plot_shape = row_len, row_len
        if not no_axes:
            figsize = 4 * plot_shape[1], 4 * plot_shape[0]
            fig, axes = plt.subplots(*plot_shape, figsize=figsize)
            plt.subplots_adjust(
                left=0,
                right=1.0,
                top=1,
                bottom=0,
                wspace=0,
                hspace=0,
            )
        else:
            axes = None
        yield vals, filtered, axes


def analyze_correlation(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    ind_var = cfg["ind_var"]
    dep_var = cfg["dep_var"]

    print(f"{ind_var} vs. {dep_var}")

    def do_group(group: pd.DataFrame, name: str) -> None:
        print(f"Group: {name}")

        kendall = True
        if kendall:
            result = kendalltau(group[ind_var], group[dep_var])
            print(
                f"correlation: {result.correlation:.2f}\t"
                f"p-value: {result.pvalue:.2f}\t"
            )
        else:
            result = linregress(group[ind_var], group[dep_var])
            print(
                f"slope: {result.slope:.2f}\t"
                f"intercept: {result.intercept:.2f}\t"
                f"rvalue: {result.rvalue:.2f}"
            )
        print()

        fig = plt.figure(figsize=(2, 1.5))
        ax = fig.add_axes([0, 0, 1, 1])
        if dep_var == "entropy":
            ax.set_ylim(-0.5, 6.5)
        elif dep_var == "steps":
            ax.set_ylim(5.5, 13)
            pass

        group.sort_values(ind_var, inplace=True)
        smoothed = gaussian_filter(group[dep_var], sigma=30)

        if ind_var != "bottleneck_size_log":
            termini = [6, 6]
        else:
            termini = [group[ind_var].min(), group[ind_var].max()]
        ax.plot(
            [group[ind_var].min(), group[ind_var].max()],
            termini,
            alpha=0.2,
            linestyle="--",
            color="gray",
        )
        ax.plot(
            [group[ind_var].min(), group[ind_var].max()],
            [0, 0],
            alpha=0.2,
            linestyle="--",
            color="gray",
        )

        ax.plot(group[ind_var], smoothed)
        alpha = min(1, 200 / len(group[ind_var]))
        ax.scatter(group[ind_var], group[dep_var], s=2.0, color="gray", alpha=alpha)

        sgn = "−" if result.correlation < 0 else "+"
        val = abs(result.correlation)
        ax.set_title(
            f"τ: {sgn}{val:.2f}",
            fontfamily="monospace",
            fontsize="x-large",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # ticks: Optional[List[Union[int, float]]] = None
        # func: Callable
        # if ind_var == "total_timesteps_log":
        #     ticks = [10000, 100_000, 1_000_000]
        #     tick_labels = ["$10^4$", "$10^5$", "$10^6$"]
        #     ax.set_xticks([np.log10(x) for x in ticks])
        #     ax.set_xticklabels(tick_labels)
        # else:
        #     if ind_var == "n_steps_log":
        #         ticks = [10, 100, 1000, 10_000]
        #         func = np.log2
        #     if ind_var == "learning_rate_log":
        #         ticks = [0.0001, 0.001, 0.01, 0.1]
        #         func = np.log10
        #     if ind_var == "bottleneck_size_log":
        #         ticks = [0x8, 0x20, 0x80]
        #         func = np.log2
        #     if ticks is not None:
        #         ax.set_xticks([func(x) for x in ticks])
        #         ax.set_xticklabels(ticks)

        # ax.set_xlim(df[ind_var].min() - 0.1, df[ind_var].max() + 0.1)
        fn = f"{ind_var}-{dep_var}-{name}".replace(".", ",")
        plt.savefig(cfg["path"] / f"{fn}.pdf", bbox_inches="tight", format="pdf")
        plt.savefig(cfg["path"] / f"{fn}.png", bbox_inches="tight", format="png")
        plt.close()

    if "groups" in cfg:
        for k, v in df.groupby(cfg["groups"]).indices.items():
            if not isinstance(k, tuple):
                kt: Tuple = (k,)
            else:
                kt = k
            name = ",".join(cfg["groups"]) + "-" + ",".join(str(x) for x in kt)
            do_group(df.iloc[v], name)
    else:
        do_group(df, "default")


def apply_transforms(x_data: np.ndarray, transforms: List, mps: Tuple) -> np.ndarray:
    for t, mp in zip(transforms, mps):
        x_data = t[1](x_data, mp)
    return x_data


TRANSFORMS = {
    "learning_rate_log": [
        ("data + x", lambda data, x: data + x, np.linspace(-10, 0, 1000)),
        ("data * x", lambda data, x: data * x, 10 ** log_range(1e-2, 1e0, 1000)),
    ]
}


def fit_metparameters(
    ind_var: str,
    x_real: np.ndarray,
    y_real: np.ndarray,
    x_model: np.ndarray,
    y_model: np.ndarray,
) -> Tuple[Tuple, float]:
    results: list = []
    tfs = TRANSFORMS[ind_var]

    for mps in product(*(t[2] for t in tfs)):
        x_model_t = apply_transforms(x_model, tfs, mps)
        idxs = np.array(
            [np.abs(x_real[i] - x_model_t).argmin() for i in range(len(x_real))]
        )
        error = ((y_model[idxs] - y_real) ** 2).mean()
        # print(f"{s:+.1f} {m:+.1f}: {np.log10(error):.1f}")
        results.append((mps, error))

    results = sorted(results, key=lambda x: x[1])
    for r in results[:20]:
        print(f"{r[0][0]:.2f} {r[0][1]:.2f} = {r[1]:.4f}")
    return results[0]


def align_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    model_data_path = cfg["model_data_path"]
    ind_var = cfg["ind_var"]
    dep_var = cfg["dep_var"]

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_ylim(-0.5, 6.5)

    ax.scatter(df[ind_var], df[dep_var], s=2.0)

    # Get model data
    data = np.genfromtxt(model_data_path, delimiter=",")
    model_xs, model_ys = data.transpose()
    if ind_var == "learning_rate_log":
        model_xs = np.log10(model_xs)
    model_plot_alpha = min(1, 0.02 * 10_000 / len(model_xs))

    mps, error = fit_metparameters(
        ind_var, df[ind_var].to_numpy(), df[dep_var].to_numpy(), model_xs, model_ys
    )
    tfs = TRANSFORMS[ind_var]
    model_xs = apply_transforms(model_xs, tfs, mps)
    print("Best transform parameters")
    for t, mp in zip(tfs, mps):
        print(f"`{t[0]}` where x={mp:.2f}")
    print(f"Error: {error:.4f}")

    ax.scatter(model_xs, model_ys, s=2.0, alpha=model_plot_alpha)

    fn = f"align-{ind_var}-{dep_var}".replace(".", ",")
    plt.savefig(cfg["path"] / f"{fn}.pdf", bbox_inches="tight", format="pdf")
    plt.savefig(cfg["path"] / f"{fn}.png", bbox_inches="tight", format="png")
    plt.close()


def preprocess_data(df: pd.DataFrame, cfg: Dict) -> None:
    if cfg.get("drop_unsuccessful", True):
        df.drop(np.flatnonzero(df["success_rate"] < 1.0), inplace=True)

    for k, v in cfg.get("drop_kv", []):
        df.drop(np.flatnonzero(df[k] == v), inplace=True)

    df["bottleneck_temperature_log"] = np.log2(df["bottleneck_temperature"])
    df["bottleneck_size"] = df["pre_bottleneck_arch"].apply(lambda x: eval(x)[-1])

    logificanda: List[Tuple[str, Callable]] = [
        ("sparsity", np.log10),
        ("learning_rate", np.log10),
        ("world_radius", np.log2),
        ("goal_radius", np.log2),
        ("bottleneck_size", np.log2),
        ("n_steps", np.log2),
        ("total_timesteps", np.log10),
        ("bottleneck_temperature", np.log2),
    ]

    for name, log in logificanda:
        df[name + "_log"] = log(df[name])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("analyses", type=str, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    for analysis in args.analyses:
        if analysis not in analysis_configs.configs:
            print(f'Analysis named "{args.analysis}" is not in analysis_configs.')
            continue

        cfg = analysis_configs.configs[analysis]

        data_path = Path(cfg["path"])
        dataframe = pd.read_csv(data_path / "data.csv").fillna("None")
        preprocess_data(dataframe, cfg)

        if cfg["type"] == "correlation":
            analyze_correlation(dataframe, cfg)
        if cfg["type"] == "align":
            align_data(dataframe, cfg)
