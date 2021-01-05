import argparse
import importlib
from pathlib import Path
from typing import Any, List
from itertools import product
import math

import pandas as pd  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore


def to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(
        escape=False,
        formatters={
            df.columns[i]: lambda x: f"${x}$"
            for i in range(len(df.columns))
            if df.dtypes[i].kind in ("i", "f")
        },
    )


def get_vector_clusters(vectors: np.ndarray, uses: np.ndarray) -> int:
    prob_thresh = 0.9
    angle_thresh = 2 * np.pi / 20
    idx = uses.argsort()[::-1]
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    for i in range(1, len(vectors)):
        if uses[idx[0:i]].sum() > prob_thresh:
            angs = angles[idx[:i]]
            proximity_mat = (
                np.abs(angs.reshape(-1, 1) - angs.reshape(1, -1)) < angle_thresh
            )
            # mat = ~np.eye(i, dtype=bool) & proximity_mat
            return sum(not proximity_mat[j, j + 1 :].any() for j in range(i))
    return len(vectors)


def add_cluster_number(df: pd.DataFrame) -> None:
    clusters = []
    for i in range(len(df)):
        vecs = df.iloc[i]["vectors"]
        uses = df.iloc[i]["usages"]
        n = get_vector_clusters(np.array(eval(vecs)), np.array(eval(uses)))
        # df.iloc[i]["clusters"] = clusters
        clusters.append(n)
    df['clusters'] = clusters


# def make_angle_plot(vectors: np.ndarray, usages: np.ndarray)
def make_snowflake_plot(df: pd.DataFrame, groups: List[str], path: Path) -> None:
    if not path.exists():
        path.mkdir()
    path = path / "snowflake"
    if not path.exists():
        path.mkdir()
    valss = product(*(df[groups[i]].unique() for i in range(len(groups))))
    for vals in valss:
        filtered = df.loc[(df[groups] == vals).all(1)]
        # TEMP
        filtered = filtered[:16]
        row_len = math.ceil(math.sqrt(len(filtered)))
        fig, axes = plt.subplots(row_len, row_len, figsize=(8, 8))
        for axis, row in zip(axes.reshape(-1), filtered.itertuples()):
            # print(f"{row.fractional:.2f}")
            # print(' '.join(f"{x:.2f}" for x in reversed(sorted(eval(row.usages)))))
            axis.axis("off")
            axis.set_xlim(-1.1, 1.1)
            axis.set_ylim(-1.1, 1.1)
            vectors = np.array(eval(row.vectors))
            uses = np.array(eval(row.usages))
            axis.set_title(f"{row.fractional:.2f} | {row.steps:.1f}")
            # clusters = get_vector_clusters(vectors, uses)
            # axis.set_title(f"{clusters:.2f}")
            for vector, use in zip(vectors, uses):
                axis.plot(
                    [0, vector[0]], [0, vector[1]], color="blue", alpha=use / max(uses)
                )
        name = "angle_summary_" + "_".join(str(v).replace(".", ",") for v in vals)
        plt.savefig(path / name)
        plt.close()


# def make_angle_plot(vectors: np.ndarray, usages: np.ndarray)
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
