from pathlib import Path

import pandas as pd  # type: ignore
import numpy as np# type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    groups = ["world_radius"]
    analyze.add_wasserstein_distance(df, path)
    grouped = df.groupby(groups)
    fields = ["argmax", "steps", "wd"]
    table = grouped.mean()[fields]
    wd_se = grouped["wd"].std() / np.sqrt(16)
    table['wd_lo'] = table['wd'] - 1.96 * wd_se
    table['wd_hi'] = table['wd'] + 1.96 * wd_se
    table = table.round(3)
    if args.figures:
        # analyze.make_snowflake_plot(df, groups, path)
        analyze.make_heatmaps(df, groups, path, plot_shape=(2,2))

    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
