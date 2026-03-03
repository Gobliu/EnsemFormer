from rdkit import Chem
from rdkit.Chem import AllChem


def smiles2pdb(smiles, pdb_path, ff, ig, max_attempts=5000, random_seed=42):
    """
    Convert a SMILES string to a PDB file with 3D coordinates.

    Args:
        smiles (str): Input SMILES string.
        pdb_path (str): Output PDB file path.
        ff (str): Force field type ('uff' or 'mmff').
        ig (bool): Ignore inter-fragment interactions during optimization.
        max_attempts (int): Max attempts for 3D embedding. Default: 5000.
        random_seed (int): Random seed for reproducibility. Default: 42.
    """
    # 1. Generate molecule from SMILES and add hydrogens
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    mol = Chem.AddHs(mol)

    # 2. Generate 3D coordinates (with random seed for reproducibility)
    AllChem.EmbedMolecule(mol, maxAttempts=max_attempts, randomSeed=random_seed)

    # 3. Force field optimization
    if ff.lower() == "uff":
        AllChem.UFFOptimizeMolecule(mol, ignoreInterfragInteractions=ig)
    elif ff.lower() == "mmff":
        AllChem.MMFFOptimizeMolecule(mol, ignoreInterfragInteractions=ig)
    else:
        raise ValueError(f"Unsupported force field: {ff}. Choose 'uff' or 'mmff'.")

    # 4. Save as PDB
    Chem.MolToPDBFile(mol, pdb_path)


if __name__ == "__main__":
    import os
    import pandas as pd

    CONFIG = {
        "ff": "uff",
        "ig": False,
        "input_csv": "Random_Split.csv",
        "output_dir": "../../../Data/For_JG",
    }

    # Create output directory
    output_dir = f"{CONFIG['output_dir']}/ff_{CONFIG['ff']}_ig_{CONFIG['ig']}"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv("Random_Split.csv")
    df = df[df["Source"] == "2015_Wang"]

    for _, row in df.iterrows():
        output_path = os.path.join(
            output_dir, f"{row['Source']}_{row['CycPeptMPDB_ID']}.pdb"
        )
        smiles2pdb(row["SMILES"], output_path, ff=CONFIG["ff"], ig=CONFIG["ig"])
