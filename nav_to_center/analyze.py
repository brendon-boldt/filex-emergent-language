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
    ind_var = cfg['ind_var']
    dep_var = cfg['dep_var']
    result = linregress(df[ind_var], df[dep_var])
    print(result)

    plt.scatter(df[ind_var], df[dep_var])
    plt.savefig(cfg['path'] / f"{ind_var}_vs_{dep_var}.png")
    plt.close()

def preprocess_data(df: pd.DataFrame) -> None:
    df['bottleneck_temperature_log'] = np.log2(df['bottleneck_temperature'])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=str)
    return parser.parse_args()

def main() -> None:
    args = get_args()

    if args.analysis not in analysis_configs.configs:
        print(f'Analysis named "{args.analysis}" is not in analysis_configs.')
        return

    cfg = analysis_configs.configs[args.analysis]

    data_path = Path(cfg['path'])
    dataframe = pd.read_csv(data_path / "data.csv").fillna("None")
    preprocess_data(dataframe)

    if cfg['type'] == 'correlation':
        analyze_correlation(dataframe, cfg)



if __name__ == "__main__":
    main()
