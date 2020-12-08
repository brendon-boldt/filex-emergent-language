import sys
import pandas as pd  # type: ignore
from pathlib import Path

def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv")
    groups = df.groupby(["discretize", "bottleneck"])
    fields = ["steps", "argmax", "fractional", "individual"]
    if args.stddev:
        op = "std"
    elif args.stderr:
        raise NotImplementedError()
    else:
        op = "mean"
    table = getattr(groups, op)()[fields].round(2)
    if args.latex:
        print(table.to_latex())
    else:
        print(table)
