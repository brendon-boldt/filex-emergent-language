from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
from scipy.stats import kendalltau  # type: ignore

import analyze  # type: ignore


def main(args, path: Path) -> None:
    df = pd.read_csv(path / "data.csv").fillna("None")
    df['run_id'] = df['path'].apply(lambda x: str(Path(x).parent).replace('/',','))
    groups = [
        "run_id",
    ]
    df = df.sort_values(
        "path", key=lambda x: x.apply(lambda x: int(x.split("-")[-1][:-3]))
    )
    analyze.add_first_step_performance(df, path)
    # analyze.add_wasserstein_distance(df, path)
    analyze.add_pareto_efficiency(df)
    # table = grouped.mean()[fields].round(3)

    # table = table.sort_values('pe')

    if args.figures:
        # analyze.make_heatmaps(df, groups, path, plot_shape=None)
        analyze.make_value_maps(df, groups,plot_shape=None)
        # analyze.make_snowflake_plot(df, groups, path, plot_shape=(3, 3))
        # analyze.make_snowflake_plot(df, groups, path)
