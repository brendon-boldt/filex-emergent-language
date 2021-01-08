from pathlib import Path

import pandas as pd  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv")
    df = df.loc[df["discretize"] == True]
    df = df.loc[(df["action_scale"] == 1) | (df["action_scale"] == 16)]
    groups = ["action_scale", "clusters"]
    analyze.add_cluster_number(df)
    df = df.set_index("clusters", append=True)
    # for i in range(df['clusters'].min(), df['clusters'].max() + 1):
    #     df[f'clusters_{i}'] = df['clusters'] == i
    grouped = df.groupby(groups)
    # fields = ["steps", "fractional", "clusters"]
    fields = ["fractional"]

    table = grouped.median()[fields].round(2)
    grouped_counts = grouped.count()["steps"]
    total_counts = df.groupby("action_scale").count()["steps"]
    table["Proportion"] = table.apply(
        lambda x: grouped_counts[x.name] / total_counts[x.name[0]], axis=1
    )
    table["Proportion"] = table["Proportion"].round(2)
    # table['clusters'] = grouped.mean()['clusters'].round(2)
    # for i in range(df['clusters'].min(), df['clusters'].max() + 1):
    #     table[f'clusters_{i}'] = grouped.mean()[f'clusters_{i}'].round(2)
    # table = table.reindex(columns=['steps', 'steps_std', 'fractional', 'fractional_std'])

    # table["steps"] = table["steps"].round(1)
    table = table.rename(
        columns={
            "steps": "Steps",
            "success_rate": "Success",
            "fractional": "$H$",
        },
        # index={1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6},
    )
    table.index.names = ["Scale", "Actions"]

    if args.figures:
        analyze.make_snowflake_plot(df, groups, path / "figs")

    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
