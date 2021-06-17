from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from scipy.stats import kendalltau # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    groups = [
            "pre_arch",
            "reward_structure",
            "gamma",
            "world_radius",
            ]
    analyze.add_first_step_performance(df, path)
    analyze.add_pareto_efficiency(df)
    grouped = df.groupby(groups)
    fields = ["argmax", 'fs_entropy', 'success_rate', "steps", "pe", 'fsp', 'fsp_pe']
    table = grouped.median()[fields].round(3)
    # table = grouped.mean()[fields].round(3)

    df['perf'] = -df.steps
    for rs in 'euclidean', 'cosine-only':
        for bn_size in 0x8, 0x10, 0x20:
            # filtered = df
            filtered = df[
                    (df['pre_arch'] == f'[32, {bn_size}]') &
                    (df['reward_structure'] == rs)
                    ]
            print(rs, bn_size)
            for tgt in 'perf', 'fsp', 'argmax', 'pe', 'fsp_pe':
            # for tgt in ['pe', 'fsp_pe']:
                gamma_only = filtered[filtered['world_radius'] == 9.]
                gamma_kt = kendalltau(gamma_only['gamma'], gamma_only[tgt])
                wr_only = filtered[filtered['gamma'] == 0.99]
                wr_kt = kendalltau(wr_only['world_radius'], wr_only[tgt])
                print(f'{tgt}', end='\t')
                print(f'gamma: {gamma_kt.correlation:+.3f}/{gamma_kt.pvalue:.2f}', end='\t')
                print(f'wr: {wr_kt.correlation:+.3f}/{wr_kt.pvalue:.2f}')
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
