import sys
import pandas as pd  # type: ignore
from pathlib import Path

import analyze  # type: ignore



def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv")
    groups = ["discretize", "env_lsize", "action_noise"]
    grouped = df.groupby(groups)
    fields = ["steps", "fractional", "linf"]
    analyze.make_angle_plot(df, groups, path / "figs")
    if args.stddev:
        op = "std"
    elif args.stderr:
        raise NotImplementedError()
    else:
        op = "median"
    table = getattr(grouped, op)()[fields].round(2)
    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
