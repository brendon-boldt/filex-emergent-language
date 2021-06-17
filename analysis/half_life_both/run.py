from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from scipy.stats import kendalltau # type: ignore

import analyze  # type: ignore

# TODO Eval on smaller environment

def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    groups = [
            "pre_arch",
            "half_life",
            ]
    analyze.add_first_step_performance(df, path)
    analyze.add_wasserstein_distance(df, path)
    analyze.add_pareto_efficiency(df)
    grouped = df.groupby(groups)
    fields = ["argmax", 'fs_entropy', 'wd', 'wd_fs', "steps", "pe", 'fsp', 'fsp_pe']
    # table = grouped.median()[fields].round(3)
    table = grouped.mean()[fields].round(3)


    df['perf'] = -df.steps
    for bn_size in [0x8]:
        # filtered = df
        filtered = df[
                (df['pre_arch'] == f'[32, {bn_size}]')
                ]
        print(bn_size)
        for tgt in ['argmax', 'perf', 'pe', 'fsp', 'fsp_pe']:
            kt = kendalltau(filtered['half_life'], filtered[tgt])
            print(f'{tgt}', end='\t')
            print(f'{kt.correlation:+.3f}/{kt.pvalue:.2f}')
        print()
    # raise SystemExit()

    # table = table.sort_values('pe')

    if args.latex:
        print(analyze.to_latex(table))
    else:
        with pd.option_context('display.max_rows', 100000, 'display.max_columns', 1000, 'display.width', 300):
            print(table)

    if args.figures:
        analyze.make_heatmaps(df, groups, path, plot_shape=(3,3))
        # analyze.make_snowflake_plot(df, groups, path, plot_shape=(3, 3))
        # analyze.make_snowflake_plot(df, groups, path)
