import argparse
import importlib
from pathlib import Path
from typing import Any, List, Optional, Tuple, Iterator
from itertools import product
import math
import pickle as pkl

from matplotlib import colors as mpcolors  # type: ignore
from matplotlib import image as mpimage  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import matplotlib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.interpolate import CubicSpline  # type: ignore
from scipy.optimize import newton  # type: ignore
import torch

import util


def to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(
        escape=False,
        formatters={
            df.columns[i]: lambda x: f"${x}$" if not pd.isna(x) else "-"
            for i in range(len(df.columns))
            if df.dtypes[i].kind in ("i", "f")
        },
    )


def get_vector_clusters(vectors: np.ndarray, uses: np.ndarray) -> int:
    prob_thresh = 0.90
    angle_thresh = 2 * np.pi / 20
    idx = uses.argsort()[::-1]
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    for i in range(1, len(vectors)):
        if uses[idx[0:i]].sum() > prob_thresh:
            angs = angles[idx[:i]]
            proximity_mat = (
                np.abs(angs.reshape(-1, 1) - angs.reshape(1, -1)) < angle_thresh
            )
            return sum(not proximity_mat[j, j + 1 :].any() for j in range(i))
    return len(vectors)


def add_cluster_number(df: pd.DataFrame) -> None:
    clusters = []
    for i in range(len(df)):
        vecs = df.iloc[i]["vectors"]
        uses = df.iloc[i]["usages"]
        n = get_vector_clusters(np.array(eval(vecs)), np.array(eval(uses)))
        clusters.append(n)
    df["clusters"] = clusters


def iter_groups(
    df: pd.DataFrame, groups: List[str], plot_shape: Optional[Tuple[int, int]]
) -> Iterator[Tuple[List, pd.DataFrame, matplotlib.axes.Axes]]:
    valss = product(*(df[groups[i]].unique() for i in range(len(groups))))
    for vals in valss:
        filtered = df.loc[(df[groups] == vals).all(1)]
        if not len(filtered):
            continue
        if plot_shape is not None:
            filtered = filtered[: plot_shape[0] * plot_shape[1]]
            # random_idxs = np.random.default_rng().choice(
            #     len(filtered),
            #     min(len(filtered), np.prod(plot_shape)),
            #     replace=False,
            # )
            # filtered = filtered.iloc[random_idxs]
        else:
            row_len = math.ceil(math.sqrt(len(filtered)))
            plot_shape = row_len, row_len
            # figsize = (8, 8)
        figsize = 4 * plot_shape[1], 4 * plot_shape[0]
        fig, axes = plt.subplots(*plot_shape, figsize=figsize)
        plt.subplots_adjust(
            left=0,
            right=1.0,
            top=1,
            bottom=0,
            wspace=0,
            hspace=0,
        )
        yield vals, filtered, axes


def make_snowflake_plot(
    df: pd.DataFrame,
    groups: List[str],
    path: Path,
    plot_shape: Optional[Tuple[int, int]] = None,
) -> None:
    if not path.exists():
        path.mkdir()
    path = path / "starfish"
    if not path.exists():
        path.mkdir()
    for vals, filtered, axes in iter_groups(df, groups, plot_shape):
        # sorted_rows = filtered.sort_values("uuid").itertuples()
        sorted_rows = filtered.itertuples()
        for axis, row in zip(axes.reshape(-1), sorted_rows):
            axis.axis("off")
            axis.set_xlim(-1.1, 1.1)
            axis.set_ylim(-1.1, 1.1)
            try:
                vectors = np.array(eval(row.vectors))
            except Exception:
                continue
            uses = np.array(eval(row.usages))
            clusters = get_vector_clusters(vectors, uses)
            axis.set_title(f"{row.pe:.3f}", loc="left", y=0.9)
            # axis.set_title(f"{clusters:.2f}")
            # axis.set_title(f"{row.fractional:.2f} - {clusters} - {row.steps:.1f}")
            for vector, use in zip(vectors, uses):
                norm = (vector ** 2).sum() ** 0.5
                vector /= max(norm, 1.0)
                axis.plot(
                    [0, vector[0]],
                    [0, vector[1]],
                    color="blue",
                    alpha=use / max(uses),
                    linewidth=8,
                    solid_capstyle="round",
                )
        name = "lexmap_" + "_".join(str(v).replace(".", ",") for v in vals)
        plt.savefig(path / name)
        plt.close()


def make_lexicon_plot(df: pd.DataFrame, groups: List[str], path: Path) -> None:
    if not path.exists():
        path.mkdir()
    valss = product(*(df[groups[i]].unique() for i in range(len(groups))))

    vis_path = path / "indiv_angle"
    if not vis_path.exists():
        vis_path.mkdir()
    top_k = 10
    for vals in valss:
        filtered = df.loc[(df[groups] == vals).all(1)]
        row_len = math.ceil(math.sqrt(len(filtered)))
        for row in filtered.itertuples():
            fig, axes = plt.subplots(2, top_k, figsize=(15, 3))
            fig.suptitle(f"{row.fractional:.2f}")
            # axis.set_title(f"{row.fractional:.2f} | {row.steps:.1f}")
            vectors = eval(row.vectors)
            uses = eval(row.usages)
            vu_pairs = sorted(zip(vectors, uses), key=lambda x: -x[1])
            for i, (vector, use) in enumerate(vu_pairs):
                if i == top_k:
                    break
                axes[1, i].axis("off")
                axes[0, i].set_xlim(-1.1, 1.1)
                axes[0, i].set_ylim(-1.1, 1.1)
                axes[0, i].xaxis.set_ticklabels([])
                axes[0, i].yaxis.set_ticklabels([])
                axes[0, i].plot([0, vector[0]], [0, vector[1]], color="blue")
                # axes[1, i].set_title(f"{use:.2f}")
                axes[0, i].set_title(f"{use:.2f}")
            name = "_".join(str(v).replace(".", ",") for v in vals) + f"_{row.Index}"
            plt.savefig(vis_path / name)
            plt.close()


def wasserstein_distance(x, p=1) -> np.ndarray:
    # https://www.stat.cmu.edu/~larry/=sml/Opt.pdf
    x = np.sort(x)
    y = np.arange(len(x)) / len(x) * 2 * np.pi - np.pi
    return (np.abs(x - y) ** p).mean() ** (1 / p)


def add_wasserstein_distance(df: pd.DataFrame, path: Path) -> None:
    df["wd"] = np.nan
    df["wd_fs"] = np.nan
    for idx, row in df.iterrows():
        traj = np.load(path / "trajectories" / (row["uuid"] + ".npz"))
        locs = traj["s"]
        angles = np.arctan2(locs[:, 0], locs[:, 1])
        df.loc[idx, "wd"] = wasserstein_distance(angles) / (2 * np.pi)

        ts = traj["t"]
        locs = locs[ts == 0]
        angles = np.arctan2(locs[:, 0], locs[:, 1])
        df.loc[idx, "wd_fs"] = wasserstein_distance(angles) / (2 * np.pi)


def h2r_fsp(h: float) -> float:
    n = 2 ** h
    return 2 * (1 - (n / np.pi) * np.sin(np.pi / n))


def r2h_fsp_approx(r: float) -> float:
    return (1 - np.log2(r)) / 1.5


def r2h_fsp(r: float) -> float:
    return newton(lambda x: h2r_fsp(x) - r, r2h_fsp_approx(r))


def add_pareto_efficiency(df: pd.DataFrame) -> None:
    df["pe"] = np.nan
    df["fsp_pe"] = np.nan
    opt_data = pd.read_csv("analysis/optimal.csv", header=None)
    opt_data[0] = np.log2(opt_data[0])
    perfs = np.array(opt_data[1])
    opt_data = opt_data[np.concatenate([[True], perfs[:-1] < perfs[1:]])]
    h2r = CubicSpline(opt_data[0], opt_data[1])
    r2h = CubicSpline(opt_data[1], opt_data[0])
    for idx, row in df.iterrows():
        r_dist = max(h2r(row["argmax"]) - -row["steps"], 0)
        h_dist = max(row["argmax"] - r2h(-row["steps"]), 0)
        if row["argmax"] < np.log2(2.9):
            df.loc[idx, "pe"] = -np.inf
        else:
            df.loc[idx, "pe"] = -np.sqrt(r_dist * h_dist)

        # first step performance
        # convert cosine similarity to euclidean distance
        # perf = 1 - row["fsp"]
        if "fsp" not in row:
            continue
        perf = 1 - row["fsp"] + 0.05
        r_dist = max(-h2r_fsp(row["argmax"]) - -perf, 0)
        h_dist = max(row["argmax"] - r2h_fsp(perf), 0)
        df.loc[idx, "fsp_pe"] = -np.sqrt(r_dist * h_dist)


def add_first_step_performance(df: pd.DataFrame, path: Path) -> None:
    df["fsp"] = np.nan
    df["fs_entropy"] = np.nan
    for idx, row in df.iterrows():
        traj = np.load(path / "trajectories" / (row["uuid"] + ".npz"))
        ts = traj["t"]
        all_acts = traj["a"]
        all_locs = traj["s"]
        locs = all_locs[ts == 0]
        acts = all_acts[ts == 0]

        probs = (
            np.expand_dims(np.unique(acts.sum(-1)), 0)
            == np.expand_dims(acts.sum(-1), -1)
        ).mean(0)
        df.loc[idx, "fs_entropy"] = -(probs * np.log2(probs)).sum()
        df.loc[idx, "fsp"] = (
            -(locs * acts).sum(-1)
            / (np.linalg.norm(locs, axis=-1) * np.linalg.norm(acts, axis=-1))
        ).mean()

        # print(row.half_life)
        # for i in range(15):
        #     acts = all_acts[(ts == i)]
        #     locs = all_locs[ts == i]
        #     cs = (
        #     -(locs * acts).sum(-1)
        #     / (np.linalg.norm(locs, axis=-1) * np.linalg.norm(acts, axis=-1))
        #     ).mean()
        #     print(f"{cs:.3f}")
        # print()


def make_heatmaps(
    df: pd.DataFrame, groups: List[str], path: Path, plot_shape: Tuple[int, int] = None
) -> None:
    for group, filtered, axes in iter_groups(df, groups, plot_shape):
        # sorted_rows = filtered#.sort_values("uuid").iterrows()
        sorted_rows = filtered.iterrows()
        for axis, (_, row) in zip(axes.reshape(-1), sorted_rows):
            axis.axis("off")
            # axis.set_title(f"{clusters:.2f}")

            # axis.set_title(row['uuid'][:10], loc='left', y=0.9)
            traj = np.load(path / "trajectories" / (row["uuid"] + ".npz"))
            locs = traj["s"]
            ts = traj["t"]
            angles = np.arctan2(locs[:, 0], locs[:, 1])

            resolution = 0x200
            disc_locs = ((locs + 1) / 2 * resolution).astype(np.int32)
            disc_locs = resolution * disc_locs[:, 1] + disc_locs[:, 0]
            disc_locs = disc_locs[(0 <= disc_locs) & (disc_locs < resolution ** 2)]
            counts = np.bincount(disc_locs, minlength=resolution ** 2).reshape(
                resolution, resolution
            )
            # scale_factor = np.sort(counts.reshape(-1))[int(0.995 * resolution ** 2)]
            scale_factor = counts.mean() * 5
            counts = (1 * counts / scale_factor).clip(0, 1)
            invertd_color_list = (1 - np.array(plt.cm.inferno.colors)).tolist()
            cmap = mpcolors.ListedColormap(invertd_color_list)
            im = axis.imshow(counts, interpolation="bicubic", cmap=cmap)

        name = "heatmap_" + "_".join(str(v).replace(".", ",") for v in group)
        fig_dir = path / "heatmaps"
        if not fig_dir.exists():
            fig_dir.mkdir()
        plt.savefig(fig_dir / name)
        plt.close()

def make_value_maps(
    df: pd.DataFrame, groups: List[str], out_path: Path, plot_shape: Tuple[int, int] = None
) -> None:
    for group, filtered, axes in iter_groups(df, groups, plot_shape):
        # sorted_rows = filtered.iterrows()
        sorted_rows = [r for r in filtered.iterrows()]
        sorted_rows = sorted_rows[:1]
        for axis, (_, row) in zip(axes.reshape(-1), sorted_rows):
            axis.axis("off")
            # axis.set_title(f"{clusters:.2f}")

            parent_path = Path(row.path).parent
            with (parent_path / "config.pkl").open("rb") as fo:
                cfg = pkl.load(fo)
            model = util.make_model(cfg)
            policy = model.policy
            policy.load_state_dict(torch.load(row.path))

            resolution = 0x80
            values = np.full([resolution] * 2, -.4, dtype=np.float32)
            for i in range(resolution):
                for j in range(resolution):
                    y = 1 - 2 * i / resolution
                    x = 2 * j / resolution - 1
                    if x ** 2 + y ** 2 > 1:
                        continue
                    obs_tensor = torch.tensor([y, x])
                    policy_out = policy(obs_tensor.unsqueeze(0), deterministic=True)
                    values[i, j] = policy_out[1].item()
            values = (values - values.min()) / (values.max() - values.min())

            invertd_color_list = (1 - np.array(plt.cm.inferno.colors)).tolist()
            cmap = mpcolors.ListedColormap(invertd_color_list)
            # im = axis.imshow(values, interpolation="bicubic", cmap=cmap)
            im = axis.imshow(values, cmap=cmap)

        name = "value_map_" + "_".join(str(v).replace(".", ",") for v in group)
        fig_dir = out_path / "value_maps"
        if not fig_dir.exists():
            fig_dir.mkdir()
        plt.savefig(fig_dir / name)
        plt.close()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str)
    parser.add_argument("--stddev", action="store_true")
    parser.add_argument("--stderr", action="store_true")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    path = Path(args.target)
    module_name = args.target.rstrip("/").replace("/", ".")
    mod: Any = importlib.import_module(f"{module_name}.run")
    mod.main(args, path)


if __name__ == "__main__":
    main()
