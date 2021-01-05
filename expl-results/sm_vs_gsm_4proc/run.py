import sys
import pandas as pd  # type: ignore
from pathlib import Path

import analyze # type: ignore

def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv")
    groups = df.groupby(["discretize", "bottleneck"])
    # fields = ["steps", "argmax", "fractional", "individual"]
    fields = ["steps", "fractional", "linf"]
    if args.stddev:
        op = "std"
    elif args.stderr:
        raise NotImplementedError()
    else:
        op = "mean"
    table = getattr(groups, op)()[fields].round(2)
    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
