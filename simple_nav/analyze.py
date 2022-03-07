import argparse
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union, Iterator, Optional, Callable
from itertools import product
import math

from scipy.stats import linregress, kendalltau  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib

from . import analysis_configs


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
            if random_idxs := True:
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


def make_snowflake_plots(
    df: pd.DataFrame,
    cfg: Dict,
) -> None:
    path = cfg["path"]
    groups = cfg["groups"]
    plot_shape = (3, 2)

    if not path.exists():
        path.mkdir()
    path = path / "starfish"
    if not path.exists():
        path.mkdir()

    for vals, filtered, axes in iter_groups(df, groups, plot_shape, no_axes=False):
        for axis, row in zip(axes.reshape(-1), filtered.itertuples()):
            axis.axis("off")
            axis.set_xlim(-1.1, 1.1)
            axis.set_ylim(-1.1, 1.1)
            try:
                vectors = np.array(eval(row.vectors))
            except Exception:
                continue
            uses = np.array(eval(row.usages))
            for vector, use in zip(vectors, uses):
                norm = (vector**2).sum() ** 0.5
                vector /= max(norm, 1.0)
                axis.plot(
                    [0, vector[0]],
                    [0, vector[1]],
                    color="blue",
                    alpha=use / max(uses),
                    linewidth=8,
                    solid_capstyle="round",
                )
        name = "lexmap_" + "_".join(str(v).replace(".", ",") for v in vals)
        plt.savefig(path / f"{name}.pdf", format="pdf")
        plt.savefig(path / f"{name}.png", format="png")
        plt.close()


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
        ax.set_ylim(-0.5, 6.5)
        ax.scatter(group[ind_var], group[dep_var], s=2.0)

        ticks: List[Union[int, float]]
        func: Callable
        if ind_var == "total_timesteps_log":
            ticks = [10000, 100_000, 1_000_000]
            tick_labels = ["$10^4$", "$10^5$", "$10^6$"]
            ax.set_xticks([np.log10(x) for x in ticks])
            ax.set_xticklabels(tick_labels)
        else:
            if ind_var == "n_steps_log":
                ticks = [10, 100, 1000, 10_000]
                func = np.log2
            if ind_var == "learning_rate_log":
                ticks = [0.0001, 0.001, 0.01, 0.1]
                func = np.log10
            if ind_var == "bottleneck_size_log":
                ticks = [0x8, 0x20, 0x80]
                func = np.log2
            ax.set_xticks([func(x) for x in ticks])
            ax.set_xticklabels(ticks)

        ax.set_xlim(df[ind_var].min() - 0.1, df[ind_var].max() + 0.1)
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


def make_histograms(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    # This function is not fully parameterized since we only generate one
    # histogram for the paper.
    dep_var = cfg["dep_var"]

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(yticklabels=[], ylabel=None, yticks=[])
    ax.set_xlabel("Entropy (bits)")

    smallest_group = min(sum(x) for x in df.groupby(cfg["groups"]).indices.values())
    n_bins = 30
    vmin = 1.8
    vmax = 5.3
    bins = [vmin + i * (vmax - vmin) / n_bins for i in range(n_bins)]
    # bins = None
    for k, v in df.groupby(cfg["groups"]).indices.items():
        if not isinstance(k, tuple):
            kt: Tuple = (k,)
        else:
            kt = k
        ax.hist(
            df.iloc[v][dep_var],
            bins=bins,
            density=True,
            alpha=0.5,
            label="Shaped" if df.iloc[v[0]]["sparsity"] < float("inf") else "No Shaped",
        )
    ax.legend(fontsize="small")
    fn = f"histogram-{dep_var}".replace(".", ",")
    plt.savefig(cfg["path"] / f"{fn}.pdf", bbox_inches="tight", format="pdf")
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
    parser.add_argument("analysis", type=str)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.analysis not in analysis_configs.configs:
        print(f'Analysis named "{args.analysis}" is not in analysis_configs.')
        return

    cfg = analysis_configs.configs[args.analysis]

    data_path = Path(cfg["path"])
    dataframe = pd.read_csv(data_path / "data.csv").fillna("None")
    preprocess_data(dataframe, cfg)

    if cfg["type"] == "correlation":
        analyze_correlation(dataframe, cfg)
    if cfg["type"] == "snowflake":
        make_snowflake_plots(dataframe, cfg)
    if cfg["type"] == "histograms":
        make_histograms(dataframe, cfg)
