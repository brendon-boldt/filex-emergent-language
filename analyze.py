import argparse
import importlib
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore


def to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(
        escape=False,
        formatters={
            df.columns[i]: lambda x: f"${x}$"
            for i in range(len(df.columns))
            if df.dtypes[i].kind in ("i", "f")
        },
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str)
    parser.add_argument("--stddev", action="store_true")
    parser.add_argument("--stderr", action="store_true")
    parser.add_argument("--latex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    path = Path(args.target)
    module_name = args.target.rstrip("/").replace("/", ".")
    mod: Any = importlib.import_module(f"{module_name}.run")
    mod.main(args, path)


if __name__ == "__main__":
    main()
