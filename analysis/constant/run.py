from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from scipy.stats import kendalltau  # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    df = df[df['success_rate'] >= 0.5]
    df = df[df['rs_multiplier'] != 1e-3]
    groups = [
        "pre_arch",
        "reward_structure",
        "rs_multiplier",
    ]
    analyze.add_first_step_performance(df, path)
    analyze.add_pareto_efficiency(df)
    grouped = df.groupby(groups)
    fields = ["argmax", "fs_entropy", "success_rate", "steps", "fsp"]
    # table = grouped.median()[fields].round(2)
    table = grouped[fields].mean().round(2)
    # table_var = (grouped[fields].std() / grouped["argmax"].count().mean() ** 0.5).round(
    # table_var = (grouped[fields].std()).round(2)
    table_var = (grouped[fields].count())

    df["perf"] = -df.steps
    for rs in ['cosine-only', 'euclidean']:
        print(rs)
        for bns in [0x10, 0x20]:
            print(bns)
            for tgt in "perf", "argmax":
                print(f"{tgt}", end="\t")
                filtered = df[
                    (df['reward_structure'] == rs) & (df['pre_arch'] == f'[32, {bns}]')
                ]
                kt = kendalltau(filtered["rs_multiplier"], filtered[tgt])
                print(f"{kt.correlation:+.3f}/{kt.pvalue:.2f}")
        print()
    # raise SystemExit()

    # table = table.sort_values('pe')

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
            print(table)
            # print(table_var)

    if args.figures:
        analyze.make_heatmaps(df, groups, path, plot_shape=None)
        # analyze.make_snowflake_plot(df, groups, path, plot_shape=(3, 3))
        analyze.make_snowflake_plot(df, groups, path)
