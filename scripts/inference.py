import json
from pathlib import Path

import click
import numpy as np
import torch

from asapdiscovery.ml.config import DatasetConfig
from asapdiscovery.ml.schema import TrainingPredictionTracker
from mtenn.config import GATModelConfig


# Function to calculate MAE
def calc_mae_df(g):
    return np.mean(np.abs(g["pred"] - g["target"]))


@click.command()
@click.option(
    "--ds-config",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Config JSON file for dataset to run inference on.",
)
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Top level model directory.",
)
@click.option(
    "--output-preds",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="File to save predictions to.",
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
    ds_config,
    model_dir,
    output_preds,
    use_epoch=None,
    use_last_epoch=False,
    use_best_epoch_loss=False,
    use_best_epoch_mae=False,
):
    # Check that all the necessary files exist
    trainer_path = model_dir / "trainer.json"
    if not trainer_path.exists():
        raise FileNotFoundError("trainer.json file not found")
    run_id_path = model_dir / "run_id"
    if not run_id_path.exists():
        raise FileNotFoundError("run_id file not found")
    weights_dir = model_dir / run_id_path.read_text()
    if not weights_dir.exists():
        raise FileNotFoundError("Weights dir not found")

    num_flags = use_last_epoch + use_best_epoch_loss + use_best_epoch_mae
    if num_flags > 0:
        pred_tracker_fn = weights_dir / "pred_tracker.json"
        if not pred_tracker_fn.exists():
            raise FileNotFoundError("pred_tracker.json file not found")

        pred_tracker = TrainingPredictionTracker(
            **json.loads(pred_tracker_fn.read_text())
        )
        full_preds_df = pred_tracker.to_plot_df(agg_compounds=False, agg_losses=True)

    # Figure out which weights file to use
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
            .apply(calc_mae_df)
            .reset_index(level=["split", "epoch"])
            .reset_index(drop=True)
            .rename(columns={0: "MAE"})
        )
        val_df = mae_df.loc[mae_df["split"] == "val", :]
        use_epoch = val_df.iloc[np.argmin(val_df["MAE"]), :]["epoch"]
    elif use_last_epoch:
        use_epoch = max(full_preds_df["epoch"])


    if use_epoch is not None:
        print(f"Using epoch {use_epoch}", flush=True)
        weights_fn = weights_dir / f"{use_epoch}.th"
        if not weights_fn.exists():
            raise FileNotFoundError("Weights file not found for specified epoch")
    else:
        weights_fn = weights_dir / "final.th"
        if not weights_fn.exists():
            all_weights_fns = [
                fn
                for fn in weights_dir.glob("*.th")
                if fn.name.strip(".th").isdecimal()
            ]
            try:
                weights_fn = max(
                    all_weights_fns, key=lambda fn: int(fn.name.strip(".th"))
                )
            except ValueError:
                raise FileNotFoundError("No weights files found in weights dir")
        print(f"Using weights file {weights_fn}", flush=True)

    # Load dataset
    print("Loading dataset", flush=True)
    ds = DatasetConfig(**json.loads(ds_config.read_text())).build()

    # Load Trainer config and get relevant info
    print("Loading Trainer kwargs", flush=True)
    trainer_kwargs = json.loads(trainer_path.read_text())
    device = torch.device(trainer_kwargs["device"])
    model_kwargs = trainer_kwargs["model_config"]

    # Won't need this anymore
    del trainer_kwargs

    # Load weights in and update model_kwargs
    print("Loading model weights", flush=True)
    weights_dict = torch.load(weights_fn, map_location=device)
    model_kwargs["model_weights"] = weights_dict

    # Build model
    print("Building model", flush=True)
    model = GATModelConfig(**model_kwargs).build()

    # Run inference
    print("Making predictions", flush=True)
    model.eval()
    with torch.no_grad():
        preds = [model(data)[0].numpy() for _, data in ds]

    # Save model predictions for submission
    print("Saving predictions", flush=True)
    preds = np.squeeze(preds)
    np.save(output_preds, preds)


if __name__ == "__main__":
    main()
