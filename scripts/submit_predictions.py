from pathlib import Path
import yaml

import click
import numpy as np
import polaris


@click.command()
@click.option(
    "--preds-spec",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Predictions YAML spec file.",
)
def main(preds_spec):
    # Load Polaris competition
    competition = polaris.load_competition("asap-discovery/antiviral-potency-2025")

    # Load submission information
    sub_spec = yaml.safe_load(preds_spec.read_text())

    # Make sure it has all required keys
    req_keys = [
        "sars_preds_file",
        "mers_preds_file",
        "prediction_name",
        "prediction_owner",
        "report_url",
    ]
    missing_keys = [k for k in req_keys if k not in sub_spec]
    if len(missing_keys) > 0:
        raise ValueError(f"Spec file missing required keys: {missing_keys}")

    # Make sure preds files exist (pop so we can use dict expansion)
    sars_preds_file = Path(sub_spec.pop("sars_preds_file"))
    if not sars_preds_file.exists():
        raise FileNotFoundError("Specified sars_preds_file not found")
    mers_preds_file = Path(sub_spec.pop("mers_preds_file"))
    if not mers_preds_file.exists():
        raise FileNotFoundError("Specified mers_preds_file not found")

    # Load predictions
    sars_preds = np.load(sars_preds_file)
    mers_preds = np.load(mers_preds_file)
    preds = {"pIC50 (SARS-CoV-2 Mpro)": sars_preds, "pIC50 (MERS-CoV Mpro)": mers_preds}

    # Submit predictions
    print(sub_spec, flush=True)
    competition.submit_predictions(predictions=preds, **sub_spec)


if __name__ == "__main__":
    main()
