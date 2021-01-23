from pathlib import Path
from typing import List

import pandas as pd  # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    groups = ["single_step", "variant", "reward_structure"]
    analyze.add_wasserstein_distance(df, path)
    grouped = df.groupby(groups)
    fields = ["argmax", "steps", "wd"]
    table = grouped.median()[fields].round(3)
    if args.figures:
        # analyze.make_heatmaps(df, groups, path, plot_shape=(2,2))
        analyze.make_snowflake_plot(df, groups, path, plot_shape=(4, 2))

    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
