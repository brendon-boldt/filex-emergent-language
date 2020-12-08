import sys
import pandas as pd  # type: ignore
from pathlib import Path

import analyze # type: ignore

def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv")
    groups = df.groupby(["discretize", "env_lsize"])
    fields = ["steps", "argmax"]

    table = groups.mean()[fields].round(2)
    table[[f"{x}_std" for x in fields]] = groups.std()[fields].round(2)
    table = table.reindex(columns=['steps', 'steps_std', 'argmax', 'argmax_std'])
    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
