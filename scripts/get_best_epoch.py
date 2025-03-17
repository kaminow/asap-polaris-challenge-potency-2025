from pathlib import Path
import json

import click
import numpy as np

from asapdiscovery.ml.schema import TrainingPredictionTracker


# Function to calculate MAE
def calc_mae_df(g):
    return np.mean(np.abs(g["pred"] - g["target"]))


@click.command()
@click.option(
    "--pred-tracker",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Location of pred_tracker.json file.",
)
def main(pred_tracker):
    pred_tracker = TrainingPredictionTracker(**json.loads(pred_tracker.read_text()))

    val_losses = pred_tracker.get_losses(agg_compounds=True, agg_losses=True)["val"]
    best_loss_epoch = np.argmin(val_losses)
    print(f"Best loss at epoch {best_loss_epoch}", flush=True)

    df = pred_tracker.to_plot_df(agg_compounds=False, agg_losses=True)
    val_idx = df["split"] == "val"
    val_maes = (
        df.loc[val_idx, :]
        .groupby(["split", "epoch"])
        .apply(calc_mae_df)
        .reset_index(level=["split", "epoch"])
        .reset_index(drop=True)
    )[0].to_numpy()
    print(val_maes, flush=True)
    best_mae_epoch = np.argmin(val_maes)
    print(f"Best MAE at epoch {best_mae_epoch}", flush=True)


if __name__ == "__main__":
    main()
