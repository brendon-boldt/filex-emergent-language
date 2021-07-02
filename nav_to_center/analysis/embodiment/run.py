from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from scipy.stats import kendalltau, pearsonr  # type: ignore
import numpy as np  # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    df = df[df["success_rate"] >= 0.5]
    df["rs_multiplier"] = np.log10(df["rs_multiplier"])
    groups = [
        "rs_multiplier",
    ]
    grouped = df.groupby(groups)
    fields = ["argmax", "success_rate", "steps"]
    # table = grouped.median()[fields].round(2)
    table = grouped[fields].mean().round(2)
    # table_var = (grouped[fields].std() / grouped["argmax"].count().mean() ** 0.5).round(
    # table_var = (grouped[fields].std()).round(2)
    table_var = grouped[fields].count()

    df["perf"] = -df.steps
    filtered = df
    for tgt in "perf", "argmax":
        print(f"{tgt}", end="\t")
        kt = kendalltau(filtered[groups[0]], filtered[tgt])
        pr = pearsonr(filtered[groups[0]], filtered[tgt])
        print(f"{kt.correlation:+.2f}/{kt.pvalue:.2f}\t", end="")
        print(f"{pr[0]:+.2f}/{pr[1]:.2f}")
    print()
    # raise SystemExit()

    # table = table.sort_values('pe')

    from matplotlib import pyplot as plt  # type: ignore

    plt.scatter(df[groups[0]], df["argmax"])
    # plt.xscale('log')
    plt.savefig(path / "group_vs_entropy.png")
    plt.close()

    if args.latex:
        print(analyze.to_latex(table))
    else:
        with pd.option_context(
            "display.max_rows",
            100000,
            "display.max_columns",
            1000,
            "display.width",
            300,
        ):
            pass
            # print(table)
            # print(table_var)

    if args.figures:
        analyze.make_heatmaps(df, groups, path, plot_shape=None)
        # analyze.make_snowflake_plot(df, groups, path, plot_shape=(3, 3))
        analyze.make_snowflake_plot(df, groups, path)
