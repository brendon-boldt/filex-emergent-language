from pathlib import Path
from typing import List

import pandas as pd  # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    df = df[df['single_step'] == False]
    groups = [
            "single_step",
            "reward_structure",
            "entropy_coef",
            "bottleneck_temperature",
            ]
    # analyze.add_wasserstein_distance(df, path)
    analyze.add_pareto_efficiency(df)
    grouped = df.groupby(groups)
    fields = ["argmax", "steps", "pe"]
    table = grouped.median()[fields].round(3)

    if args.figures:
        analyze.make_heatmaps(df, groups, path, plot_shape=(3,3))
        analyze.make_snowflake_plot(df, groups, path, plot_shape=(3, 3))

    table = table.sort_values('pe')

    if args.latex:
        print(analyze.to_latex(table))
    else:
        with pd.option_context('display.max_rows', 100000, 'display.max_columns', 1000, 'display.width', 300):
            print(table)
