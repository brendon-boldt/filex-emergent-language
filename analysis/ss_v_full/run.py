from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from scipy import stats  # type: ignore

import analyze  # type: ignore


def cdf_diff(x, n) -> np.ndarray:
    return sum(
        np.abs(i / n - (x < (2 * np.pi * i / n - np.pi)).mean()) for i in range(1, n)
    ) / (n - 1)

def wasserstein_distance(x, p=1) -> np.ndarray:
    # https://www.stat.cmu.edu/~larry/=sml/Opt.pdf
    x = np.sort(x)
    y = np.arange(len(x)) / len(x) * 2 * np.pi - np.pi    
    return (np.abs(x-y) ** p).mean() ** (1/p)



def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    groups = ["single_step", "variant", "reward_structure"]

    df["wd1"] = np.nan
    df["wd2"] = np.nan
    for idx, row in df.iterrows():
        # print(f"{str(row[groups].tolist()):<40}", end="")
        traj = np.load(path / "trajectories" / (row["uuid"] + ".npz"))
        locs = traj["s"]
        ts = traj["t"]
        norms = (locs ** 2).sum(-1) ** 0.5
        angles = np.arctan2(locs[:, 0], locs[:, 1])
        angles = angles[ts > 5]
        # ks_res = stats.kstest(angles, stats.uniform(-np.pi, 2 * np.pi).cdf)
        # df.loc[idx, "ks"] = ks_res.statistic
        df.loc[idx, "wd1"] = cdf_diff(angles, 1000)
        df.loc[idx, "wd2"] = wasserstein_distance(angles) / (2 *np.pi)

        # print((np.histogram(angles, 12)[0] / len(angles)).round(2))
        # val = s33
        # print(f"{val:+.3f}")

    grouped = df.groupby(groups)
    fields = ["argmax", "steps", "wd1", "wd2"]
    table = grouped.median()[fields].round(3)
    if args.figures:
        analyze.make_snowflake_plot(df, groups, path / "figs")

    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
