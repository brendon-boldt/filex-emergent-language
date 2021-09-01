import argparse
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union, Iterator, Optional, Callable
from itertools import product
import math

from scipy.stats import linregress  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib

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


def make_snowflake_plots(
    df: pd.DataFrame,
    cfg: Dict,
) -> None:
    path = cfg['path']
    groups = cfg['groups']
    plot_shape = None

    if not path.exists():
        path.mkdir()
    path = path / "starfish"
    if not path.exists():
        path.mkdir()
    for vals, filtered, axes in iter_groups(df, groups, plot_shape):
        # sorted_rows = filtered.sort_values("uuid").itertuples()
        sorted_rows = filtered.itertuples()
        for axis, row in zip(axes.reshape(-1), sorted_rows):
            axis.axis("off")
            axis.set_xlim(-1.1, 1.1)
            axis.set_ylim(-1.1, 1.1)
            try:
                vectors = np.array(eval(row.vectors))
            except Exception:
                continue
            uses = np.array(eval(row.usages))
            axis.set_title(f"{row.success_rate:.2f} - {row.entropy:.3f}", loc="left", y=0.9)
            for vector, use in zip(vectors, uses):
                norm = (vector ** 2).sum() ** 0.5
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
        plt.savefig(path / name)
        plt.close()


def analyze_correlation(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    ind_var = cfg["ind_var"]
    dep_var = cfg["dep_var"]

    print(f"{ind_var} vs. {dep_var}")

    def do_group(df: pd.DataFrame, name: str) -> None:
        result = linregress(df[ind_var], df[dep_var])
        print(f"Group: {name}")
        # TODO Make a LaTeX and normal version
        print(
            f"slope: {result.slope:.2f}\t"
            f"intercept: {result.intercept:.2f}\t"
            f"rvalue: {result.rvalue:.2f}"
        )
        print()

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(df[ind_var], df[dep_var], s=0.5)
        ticks: List[Union[int, float]]
        if dep_var == "entropy":
            # ax.set_ylim(1.5, 5.1)
            # ax.set_yticks([2, 3, 4, 5])
            ax.set_ylim(1.5, 7.1)
            pass
        if ind_var == "bottleneck_size_log":
            ax.set_ylim(2.9, 8.1)
            ax.plot([3, 8], [3, 8], color="gray", linewidth=1.0)
            ax.set_yticks([3, 4, 5, 6, 7, 8])
            ticks = [8, 32, 128, 512]
            ax.set_xticks([np.log2(x) for x in ticks])
            ax.set_xticklabels(ticks)
        if ind_var == "world_radius_log":
            pass
            # ticks = [2, 4, 8, 16]
            # ax.set_xticks([np.log2(x) for x in ticks])
            # ax.set_xticklabels(ticks)
            # ax.set_xlim(0.8, np.log2(22))
            # ax.set_yticks([3, 3.5, 4, 4.5])
            # ax.set_ylim(2.8, 4.9)
            # ax.set_ylabel("Entropy (bits)")
            # ax.set_xlabel("World Radius")
        if ind_var == "learning_rate_log":
            ticks = [1e-4, 1e-3, 1e-2, 1e-1]
            ax.set_xticks([np.log10(x) for x in ticks])
            ax.set_xticklabels(ticks)
            ax.set_xlim(-4.2, -0.8)
        if ind_var == "bottleneck_temperature_log":
            ticks = [0.5, 0.75, 1, 1.5, 2]
            ax.set_xticks([np.log2(x) for x in ticks])
            ax.set_xticklabels(ticks)
        if ind_var == "sparsity_log":
            # ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
            # ax.set_yticks([3, 4, 5, 6, 7, 8])
            ax.set_ylim(1.9, 7.1)
            ticks = [1, 100, 10_000]
            ax.set_xticks([np.log10(x) for x in ticks])
            ax.set_xticklabels(ticks)
        fn = f"{ind_var}-{dep_var}-{name}".replace(".", ",")
        plt.savefig(cfg["path"] / f"{fn}.pdf", bbox_inches="tight", format="pdf")
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


def preprocess_data(df: pd.DataFrame, cfg: Dict) -> None:
    if cfg.get('drop_unsuccessful', True):
        df.drop(np.flatnonzero(df["success_rate"] < 1.0), inplace=True)
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
        df[name + '_log'] = log(df[name])


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
