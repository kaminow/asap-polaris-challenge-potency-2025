from pathlib import Path
import json

import click
import numpy as np

from asapdiscovery.ml.schema import TrainingPredictionTracker


@click.command()
@click.option(
    "--pred-tracker",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Location of pred_tracker.json file.",
)
def main(pred_tracker):
    pred_tracker = TrainingPredictionTracker(**json.loads(pred_tracker.read_text()))
    val_losses = pred_tracker.get_losses(True, True)["val"]
    use_epoch = np.argmin(val_losses)
    print(f"Best loss at epoch {use_epoch}", flush=True)


if __name__ == "__main__":
    main()
