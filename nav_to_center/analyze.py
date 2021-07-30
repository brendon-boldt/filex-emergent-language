import argparse
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union

from scipy.stats import linregress  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from . import analysis_configs


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
            # ax.set_ylabel("Entropy (bits)")
            ax.set_ylim(1.5, 5.1)
            ax.set_yticks([2, 3, 4, 5])
        if ind_var == "bottleneck_size_log":
            ax.set_ylim(2.9, 8.1)
            ax.plot([3, 8], [3, 8], color="gray", linewidth=1.0)
            ax.set_yticks([3, 4, 5, 6, 7, 8])
            ticks = [8, 32, 128, 512]
            ax.set_xticks([np.log2(x) for x in ticks])
            ax.set_xticklabels(ticks)
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
        if ind_var == "sparsity_log":
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
    df["sparsity_log"] = np.log10(df["sparsity"])
    df["learning_rate_log"] = np.log10(df["learning_rate"])
    df["bottleneck_size"] = df["pre_bottleneck_arch"].apply(lambda x: eval(x)[-1])
    df["bottleneck_size_log"] = np.log2(df["bottleneck_size"])


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
