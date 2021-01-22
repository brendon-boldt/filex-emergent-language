from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from matplotlib import image as mpimage  # type: ignore

import analyze  # type: ignore


def wasserstein_distance(x, p=1) -> np.ndarray:
    # https://www.stat.cmu.edu/~larry/=sml/Opt.pdf
    x = np.sort(x)
    y = np.arange(len(x)) / len(x) * 2 * np.pi - np.pi
    return (np.abs(x - y) ** p).mean() ** (1 / p)


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    groups = ["single_step", "variant", "reward_structure"]

    df["wd"] = np.nan
    # TODO Abstract this into analyze.py
    for idx, row in df.iterrows():
        # print(f"{str(row[groups].tolist()):<40}", end="")
        traj = np.load(path / "trajectories" / (row["uuid"] + ".npz"))
        locs = traj["s"]
        ts = traj["t"]
        norms = (locs ** 2).sum(-1) ** 0.5
        angles = np.arctan2(locs[:, 0], locs[:, 1])
        angles = angles[ts > 5]
        df.loc[idx, "wd"] = wasserstein_distance(angles) / (2 * np.pi)

        fig_path = path / "figs"
        resolution = 0x100
        # image = np.full([resolution] * 2 + [3], 255, dtype=np.uint8)
        # counts = np.zeros([resolution]* 2, dtype=np.int32)
        disc_locs = ((locs + 1) / 2 * resolution).round().astype(np.int32)
        disc_locs = resolution * disc_locs[:, 1] + disc_locs[:, 0]
        counts = np.bincount(disc_locs, minlength=resolution ** 2).reshape(
            resolution, resolution
        )
        scale_factor = np.sort(counts.reshape(-1))[int(0.995 * resolution ** 2)]
        counts = (1 * counts / scale_factor).clip(0, 1)
        # counts = np.expand_dims(counts, -1).repeat(3, -1)
        # image[disc_locs[:, 0], disc_locs[:, 1], :] = 0
        name = (
            "_".join(str(x) for x in row[groups].tolist()) + "_" + row["uuid"] + ".png"
        )
        plt.imshow(counts, interpolation='bicubic', cmap='viridis')
        plt.savefig(fig_path / name)
        plt.close()

    grouped = df.groupby(groups)
    fields = ["argmax", "steps", "wd"]
    table = grouped.median()[fields].round(3)
    if args.figures:
        analyze.make_snowflake_plot(df, groups, path / "figs")

    if args.latex:
        print(analyze.to_latex(table))
    else:
        print(table)
