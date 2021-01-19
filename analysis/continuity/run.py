from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

import analyze  # type: ignore


def cluster_table(args, df: pd.DataFrame, proporition=False) -> None:
    df = df.loc[df['discretize'] == True].copy()
    groups = ["action_scale", 'clusters']
    analyze.add_cluster_number(df)
    for i in range(df["clusters"].min(), df["clusters"].max() + 1):
        df[f"clusters_{i}"] = df["clusters"] == i
    grouped = df.groupby(groups)
    fields: List[str] = ['steps']

    if proporition:
        # TODO 60 should not be hardcoded in here
        table = (grouped.count()[fields] / 60).round(2)
    else:
        table = grouped.mean()[fields].round(2)
    # table["clusters"] = grouped.mean()["clusters"].round(2)
    for i in range(df["clusters"].min(), df["clusters"].max() + 1):
        # table[f"clusters_{i}"] = grouped.mean()[f"clusters_{i}"].round(2)
        pass

    table = table.rename(
        columns={
            "steps": "Steps",
            "fractional": "$H$",
        },
        # index={1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6},
    )
    table.index.names = ["Scale", "Actions"]
    table = table.unstack("Actions")

    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv")
    cluster_table(args, df.copy())
    cluster_table(args, df.copy(), True)
    groups = ["discretize", "action_scale"]
    grouped = df.groupby(groups)
    fields = ["steps", "fractional"]

    table = grouped.median()[fields].round(2)

    table["steps"] = table["steps"].round(1)
    table = table.rename(
        columns={
            "steps": "Steps",
            "success_rate": "Success",
            "fractional": "$H$",
        },
    )
    table = table.rename(
        index={False: "No", True: "Yes"},
        level=0,
        # index={1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6},
    )
    table.index.names = ["One-Hot", "Scale"]

    if args.figures:
        analyze.make_snowflake_plot(df, groups, path / "figs", variant="columns")

    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
