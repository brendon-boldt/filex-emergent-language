from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from scipy.stats import kendalltau, pearsonr  # type: ignore
import numpy as np  # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    df = df[df["success_rate"] >= 0.5]
    df["learning_rate"] = np.log10(df["learning_rate"])
    groups = [
        "learning_rate",
    ]
    grouped = df.groupby(groups)

    df["perf"] = -df.steps
    filtered = df
    for tgt in "perf", "argmax":
        print(f"{tgt}", end="\t")
        kt = kendalltau(filtered[groups[0]], filtered[tgt])
        pr = pearsonr(filtered[groups[0]], filtered[tgt])
        print(f"{kt.correlation:+.2f}/{kt.pvalue:.2f}\t", end="")
        print(f"{pr[0]:+.2f}/{pr[1]:.2f}")
    print()

    from matplotlib import pyplot as plt  # type: ignore

    plt.scatter(df[groups[0]], df["argmax"])
    plt.savefig(path / "group_vs_entropy.png")
    plt.close()
