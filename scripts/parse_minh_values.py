"""
Want to unify the formatting of the Minh calculated data to match how it's expected to
be so we can use the standard asap-cli data pipeline.
"""

from pathlib import Path

import numpy as np
import pandas

in_csv = Path(
    "/data1/choderaj/kaminowb/polaris_challenge_2025/raw_data/potency/"
    "MERS-CoV-Mpro_potencies_CONFIDENTIAL.csv"
)
out_csv = Path(
    "/data1/choderaj/kaminowb/polaris_challenge_2025/raw_data/potency/"
    "MERS-CoV-Mpro_potencies_CONFIDENTIAL_minh_assay_fixed.csv"
)

df = pandas.read_csv(in_csv)

# Columns to load data from
in_assay_name = "MERS-CoV-MPro_fluorescence-dose-response_weizmann"
in_pic50_col = f"{in_assay_name}: Minh_Protease_MERS_Mpro_pIC50 (calc) (uM)"
in_stdev_col = (
    f"{in_assay_name}: Minh_Protease_MERS_Mpro_pIC50 (calc) (uM) Standard Deviation (±)"
)
in_count_col = f"{in_assay_name}: Minh_Protease_MERS_Mpro_pIC50 (calc) (uM) Count"
in_curve_class_col = f"{in_assay_name}: Curve class"
in_hill_slope_col = f"{in_assay_name}: Hill slope"

# Columns to save new data to
out_assay_name = "MERS-CoV-MPro_fluorescence-dose-response_weizmann_minh"
out_ic50_col = f"{out_assay_name}: IC50 (µM)"
out_ci_low_col = f"{out_assay_name}: IC50 CI (Lower) (µM)"
out_ci_high_col = f"{out_assay_name}: IC50 CI (Upper) (µM)"
out_curve_class_col = f"{out_assay_name}: Curve class"
out_hill_slope_col = f"{out_assay_name}: Hill slope"

# Calculate IC50s from Minh-calculated pIC50s (for compatibility)
out_ic50s = np.power(10, -df[in_pic50_col].to_numpy()) / (10e-6)

# Reverse calculate 95% CI from standard deviations
# CI = X +- Z * (standard dev / sqrt(sample size))
# for 95% CI, Z = 1.96
ci_pic50 = 1.96 * df[in_stdev_col] / np.sqrt(df[in_count_col])
# Convert pIC50 units -> IC50 (uM)
ci_ic50 = np.power(10, -ci_pic50) / (10e-6)

df[out_ic50_col] = out_ic50s
# df[out_ci_low_col] = out_ic50s - ci_ic50
df[out_ci_low_col] = np.nan
# df[out_ci_high_col] = out_ic50s + ci_ic50
df[out_ci_high_col] = np.nan
df[out_curve_class_col] = df[in_curve_class_col]
df[out_hill_slope_col] = df[in_hill_slope_col]
df.to_csv(out_csv, index=False)
