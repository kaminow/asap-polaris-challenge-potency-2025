from pathlib import Path

from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.ml.config import DatasetConfig

# Load SMILES
smi_path = Path(
    "/data1/choderaj/kaminowb/polaris_challenge_2025/raw_data/potency/test_smi.smi"
)
test_smi = smi_path.read_text().split("\n")

# Build Ligands
ligs = [
    Ligand.from_smiles(smi, compound_name=f"test{i}") for i, smi in enumerate(test_smi)
]

# Construct ds config and build ds
ds_path = Path("/data1/choderaj/kaminowb/polaris_challenge_2025/ml_datasets")
ds_config = DatasetConfig(
    ds_type="graph", input_data=ligs, cache_file=(ds_path / "test_data_gat_ds.pkl")
)
(ds_path / "test_data_gat_config.json").write_text(ds_config.json())
ds_config.build()
