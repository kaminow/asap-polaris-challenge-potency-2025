from pathlib import Path

import polaris

competition = polaris.load_competition("asap-discovery/antiviral-potency-2025")

# Download and cache competition data
competition.cache()

# Only care about the test split (stored as list of CXSMILES)
_, test_smi = competition.get_train_test_split()

# Save SMILES to output file
out_fn = Path(
    "/data1/choderaj/kaminowb/polaris_challenge_2025/raw_data/potency/test_smi.smi"
)
out_fn.write_text("\n".join(list(test_smi)))
