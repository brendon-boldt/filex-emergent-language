import argparse
import importlib
from pathlib import Path
from typing import Any, List, Optional, Tuple, Iterator, Dict
from itertools import product
import math

from scipy.stats import linregress  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import matplotlib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from . import analysis_configs


def iter_groups(
    df: pd.DataFrame, groups: List[str], plot_shape: Optional[Tuple[int, int]]
) -> Iterator[Tuple[List, pd.DataFrame, matplotlib.axes.Axes]]:
    valss = product(*(df[groups[i]].unique() for i in range(len(groups))))
    for vals in valss:
        filtered = df.loc[(df[groups] == vals).all(1)]
        if not len(filtered):
            continue
        if plot_shape is not None:
            filtered = filtered[: plot_shape[0] * plot_shape[1]]
            # random_idxs = np.random.default_rng().choice(
            #     len(filtered),
            #     min(len(filtered), np.prod(plot_shape)),
            #     replace=False,
            # )
            # filtered = filtered.iloc[random_idxs]
        else:
            row_len = math.ceil(math.sqrt(len(filtered)))
            plot_shape = row_len, row_len
            # figsize = (8, 8)
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
        yield vals, filtered, axes


def analyze_correlation(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    ind_var = cfg["ind_var"]
    dep_var = cfg["dep_var"]

    print(f"{ind_var} vs. {dep_var}")

    def do_group(df: pd.DataFrame, name: str) -> None:
        result = linregress(df[ind_var], df[dep_var])
        print(f"Group: {name}")
        # TODO Make a LaTeX and normal version
        for x in ["slope", "intercept", "rvalue"]:
            print(f"{getattr(result, x):.2f}", end="\t")
        print()

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(df[ind_var], df[dep_var], s=0.5)
        if dep_var == "entropy":
            # ax.set_ylabel("Entropy (bits)")
            ax.set_ylim(1.5, 5.1)
            ax.set_yticks([2, 3, 4, 5])
        if ind_var == "lexicon_size_log":
            ax.set_ylim(2.9, 8.1)
            ax.plot([3, 8], [3, 8], color="gray", linewidth=1.0)
            ax.set_yticks([3, 4, 5, 6, 7, 8])
        if ind_var == "learning_rate_log":
            ax.set_xlabel("Bottleneck Temperature")
            ticks = [1e-4, 1e-3, 1e-2, 1e-1]
            ax.set_xticks([np.log10(x) for x in ticks])
            ax.set_xticklabels(ticks)
            ax.set_xlim(-4.2, -0.8)
        if ind_var == "bottleneck_temperature_log":
            # ax.set_xlabel("Bottleneck Temperature")
            ticks = [0.5, 0.75, 1, 1.5, 2]
            ax.set_xticks([np.log2(x) for x in ticks])
            ax.set_xticklabels(ticks)
        if ind_var == "rs_multiplier_log":
            # ax.set_xlabel("Reward Shaping Multiplier")
            # ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
            ticks = [1e-4, 1e-2, 1e0]
            ax.set_xticks([np.log10(x) for x in ticks])
            ax.set_xticklabels(ticks)
        raw_fn = f"{ind_var}-{dep_var}-{name}"
        plt.savefig(
            cfg["path"] / raw_fn.replace(".", ","),
            bbox_inches="tight",
        )
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


def preprocess_data(df: pd.DataFrame) -> None:
    df.drop(np.flatnonzero(df["success_rate"] < 1.0), inplace=True)
    df["bottleneck_temperature_log"] = np.log2(df["bottleneck_temperature"])
    df["rs_multiplier_log"] = np.log10(df["rs_multiplier"])
    df["learning_rate_log"] = np.log10(df["learning_rate"])
    df["lexicon_size"] = df["pre_arch"].apply(lambda x: eval(x)[-1])
    df["lexicon_size_log"] = np.log2(df["lexicon_size"])


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
    preprocess_data(dataframe)

    if cfg["type"] == "correlation":
        analyze_correlation(dataframe, cfg)
