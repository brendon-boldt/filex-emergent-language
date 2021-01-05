import sys
import pandas as pd  # type: ignore
from pathlib import Path

import analyze  # type: ignore


def sort_key(s: pd.Series) -> pd.Series:
    if type(s[0]) == bool:
        return s
    else:
        return s.apply(lambda x: eval(x)[-1])


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv")
    breakpoint()
    groups = ["env_lsize", "discretize", "pre_arch"]
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
    table = table.sort_values(by=["discretize", "pre_arch"], key=sort_key)
    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
