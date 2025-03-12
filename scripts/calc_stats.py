from functools import partial
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path

import click
import numpy as np
import pandas
from scipy.stats import bootstrap, kendalltau, spearmanr

from asapdiscovery.ml.schema import TrainingPredictionTracker


# Function to calculate MAE
def calc_mae_df(g):
    return np.mean(np.abs(g["pred"] - g["target"]))


# Function to calculate a statistic (for multiprocessing)
def calc_one_stat(stat_func, target_vals, preds):
    val = stat_func(target_vals, preds)
    try:
        conf_interval = bootstrap(
            (target_vals, preds),
            statistic=lambda target, pred: stat_func(target, pred),
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
    except ValueError as e:
        print(target_vals, preds, flush=True)
        raise e

    print("finished", stat_func, flush=True)
    return val, conf_interval


# Different stat functions
def calc_mae(target_vals, preds):
    return np.abs(target_vals - preds).mean()


def calc_rmse(target_vals, preds):
    return np.sqrt(np.power(target_vals - preds, 2).mean())


def calc_spearmanr(target_vals, preds):
    return spearmanr(target_vals, preds).statistic


def calc_kendalltau(target_vals, preds):
    return kendalltau(target_vals, preds).statistic


stats_funcs = [calc_mae, calc_rmse, calc_spearmanr, calc_kendalltau]


@click.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Top level model directory.",
)
@click.option(
    "--output-csv",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="File to save stats to.",
)
@click.option(
    "--use-epoch",
    type=int,
    help=(
        "Which epoch weights file to use. If not specified, final.th will be used. "
        "If that does not exist, the most recent epoch file will be used."
    ),
)
@click.option(
    "--use-last-epoch",
    is_flag=True,
    help="Load loss values from pred_tracker and use last epoch.",
)
@click.option(
    "--use-best-epoch-loss",
    is_flag=True,
    help="Load loss values from pred_tracker and use epoch with lowest val loss.",
)
@click.option(
    "--use-best-epoch-mae",
    is_flag=True,
    help="Load loss values from pred_tracker and use epoch with lowest val MAE.",
)
def main(
    model_dir,
    output_csv,
    use_epoch=None,
    use_last_epoch=False,
    use_best_epoch_loss=False,
    use_best_epoch_mae=False,
):
    run_id_path = model_dir / "run_id"
    if not run_id_path.exists():
        raise FileNotFoundError("run_id file not found")
    pred_tracker_fn = model_dir / run_id_path.read_text() / "pred_tracker.json"
    if not pred_tracker_fn.exists():
        raise FileNotFoundError("pred_tracker.json file not found")
    pred_tracker = TrainingPredictionTracker(**json.loads(pred_tracker_fn.read_text()))

    full_preds_df = pred_tracker.to_plot_df(agg_compounds=False, agg_losses=True)

    num_flags = use_last_epoch + use_best_epoch_loss + use_best_epoch_mae
    if use_epoch is not None:
        if num_flags > 0:
            print(
                "--use-epoch was specified in addition to some number of flags, "
                "ignoring flags in favor of specified epoch.",
                flush=True,
            )
    elif num_flags == 0:
        print("No epoch specified to use, using final epoch.", flush=True)
        use_epoch = max(full_preds_df["epoch"])
    elif num_flags > 1:
        raise ValueError("Too many flags specified.")
    elif use_best_epoch_loss:
        epoch_df = pred_tracker.to_plot_df(agg_compounds=True, agg_losses=True)
        val_df = epoch_df.loc[epoch_df["split"] == "val", :]
        use_epoch = val_df.iloc[np.argmin(val_df["loss"]), :]["epoch"]
    elif use_best_epoch_mae:
        mae_df = (
            full_preds_df.groupby(["split", "epoch"])
            .apply(calc_mae)
            .reset_index(level=["split", "epoch"])
            .reset_index(drop=True)
            .rename(columns={0: "MAE"})
        )
        val_df = mae_df.loc[mae_df["split"] == "val", :]
        use_epoch = val_df.iloc[np.argmin(val_df["MAE"]), :]["epoch"]
    elif use_last_epoch:
        use_epoch = max(full_preds_df["epoch"])

    # Index for the correct epoch and val split
    print(f"Using epoch {use_epoch}", flush=True)
    use_idx = (full_preds_df["epoch"] == use_epoch) & (full_preds_df["split"] == "val")
    use_df = full_preds_df.loc[use_idx, :]

    target_vals = use_df["target"].values
    preds = use_df["pred"].values

    mp_func = partial(calc_one_stat, target_vals=target_vals, preds=preds)

    # Values and low/high bounds of 95% CIs for all stats
    stat_names = []
    stat_vals = []
    stat_95ci_lows = []
    stat_95ci_highs = []
    with mp.Pool(processes=4) as pool:
        stat_res = pool.map(mp_func, stats_funcs)
    # stat_res = [mp_func(f) for f in stats_funcs]
    for stat_name, (val, conf_interval) in zip(
        ["MAE", "RMSE", r"Spearman's $\rho$", r"Kendall's $\tau$"], stat_res
    ):
        stat_names.append(stat_name)
        stat_vals.append(val)
        stat_95ci_lows.append(conf_interval.low)
        stat_95ci_highs.append(conf_interval.high)

    stats_dict = {
        "Num Compounds": len(preds),
        "Statistic": stat_names,
        "Value": stat_vals,
        "95ci_low": stat_95ci_lows,
        "95ci_high": stat_95ci_highs,
    }
    stats_df = pandas.DataFrame(stats_dict)
    stats_df.to_csv(output_csv)


if __name__ == "__main__":
    main()
