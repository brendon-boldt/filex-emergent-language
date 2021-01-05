import sys
from pathlib import Path

import pandas as pd  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv")
    groups = ["discretize", "env_lsize"]
    grouped = df.groupby(groups)
    fields = ["steps", "fractional"]

    analyze.make_snowflake_plot(df, groups, path / "figs")

    # table = grouped.mean()[fields].round(2)
    table = grouped.median()[fields].round(2)
    table[[f"{x}_std" for x in fields]] = grouped.std()[fields].round(2)
    # table[[f"{x}_min" for x in fields]] = grouped.min()[fields].round(2)
    # table[[f"{x}_max" for x in fields]] = grouped.max()[fields].round(2)
    # table = table.reindex(columns=['steps', 'steps_std', 'fractional', 'fractional_std'])

    table["steps"] = table["steps"].round(1)
    table = table.rename(
        columns={
            "steps": "Steps",
            "fractional": "$H$",
            "discretize": "One-Hot" ,
        },
        index={
            "discretize": "One-Hot" ,
            }
    )

    fig, axes = plt.subplots(nrows=4, figsize=(6, 6))
    for i, p in enumerate(range(4, 8)):
        # ax = plt.axes(label=f"{i**2}")
        ax = axes[i]
        ax.set_xlim(xmin=10, xmax=32)
        ax.set_ylim(ymax=25)
        # ax.set_xbound(lower=10,x_max=35)
        data = df.loc[(df.env_lsize == p) & (df.discretize == False)]["steps"]
        ax.hist(data)
    plt.savefig(f"figs/4567.png")

    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
