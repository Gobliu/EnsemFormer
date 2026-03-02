import os
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from PDB_checker import pdb_checker


def smiles2pdb(smiles: str, pdb_path: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES:", smiles)
    mol_h = Chem.AddHs(mol)  # include implicit hydrogens
    AllChem.EmbedMolecule(mol_h)
    pdb_block = Chem.MolToPDBBlock(mol_h)
    with open(pdb_path, "w") as f:
        f.write(pdb_block)


def check_move_rename(csv_path, match_pdb_dir, mismatch_pdb_dir):
    df = pd.read_csv(csv_path, low_memory=False)

    mismatches = []
    missing_files = []
    for idx, row in df.iterrows():
        pdb_path = f"{row['log_path'][:-4]}_1st_cluster.pdb"
        if not os.path.exists(pdb_path):
            pdb_path = f"{row['log_path'][:-4]}_1st_frame.pdb"
        if not os.path.exists(pdb_path):
            # print('File does not exist', row['log_path'])
            missing_files.append(row["CycPeptMPDB_ID"])
            continue

        match_flag = pdb_checker(pdb_path, row["SMILES"])

        if match_flag:
            source = row["Source"].strip()
            filename = f"{source}_{row['CycPeptMPDB_ID']}.pdb"
            dest_path = os.path.join(match_pdb_dir.strip(), filename)
            print(f"Moving {pdb_path} -> {dest_path}")
            shutil.move(pdb_path, dest_path)
        else:
            print(f"!!! Mismatch {row['log_path']}")
            mismatches.append((idx, row["SMILES"]))
            dest_path = os.path.join(mismatch_pdb_dir.strip(), pdb_path.split("/")[-1])
            shutil.copy2(pdb_path, dest_path)
            gen_path = dest_path[:-16] + "_rdkit.pdb"
            smiles2pdb(row["SMILES"], gen_path)

    print(f"Total mismatches: {len(mismatches)}")


if __name__ == "__main__":
    check_move_rename(
        csv_path="Peptide_with_pdb.csv",
        match_pdb_dir="/home/liuwei/GitHub/Data/Hexene",
        mismatch_pdb_dir="/home/liuwei/GitHub/Data/mismatch",
    )
