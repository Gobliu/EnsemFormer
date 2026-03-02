# Adapted from CPMP.data.pampa_uff_ig_true

import pandas as pd
import torch


from LoadDataFromPDB import load_data_from_pdb

charge_dict = {"H": 1, "C": 6, "N": 7, "O": 8}
one_hot_dict = {"H": 0, "C": 1, "N": 2, "O": 3}


def read_single_pdb(pdb_path, max_atom):
    with open(pdb_path, "r") as pdb_file:
        pdb = [line for line in pdb_file.readlines() if line.startswith("ATOM")]
        num_atoms = len(pdb)
        charges = torch.zeros((max_atom,), dtype=torch.int)
        positions = torch.zeros((max_atom, 3))
        one_hot = torch.zeros((max_atom, len(one_hot_dict)), dtype=torch.bool)

        for j in range(num_atoms):
            line = pdb[j].strip().split()
            atom_type = line[2][0]

            if atom_type not in charge_dict:
                raise ValueError(f"Unknown atom type: {atom_type}")

            charges[i] = charge_dict[atom_type]
            positions[i, :] = torch.tensor([float(k) for k in line[5:8]])
            one_hot[i, one_hot_dict[atom_type]] = True

    return num_atoms, charges, positions, one_hot


def check_max_atom(ip_csv_path, pdb_dir):
    column_list = ["SMILES", "PAMPA", "CycPeptMPDB_ID", "Source"]
    print(column_list)
    df = pd.read_csv(ip_csv_path, low_memory=False)[column_list]
    for row in df.iterrows():
        pdb_path = f"{pdb_dir}/{row.Source}_{row.CycPeptMPDB_ID}.pdb"


def data_processor_from_pdb(ip_csv_path, pdb_dir, remove_h, split_seed):
    column_list = ["SMILES", "PAMPA", "CycPeptMPDB_ID", f"split{split_seed}", "Source"]
    print(column_list)
    df = pd.read_csv(ip_csv_path, low_memory=False)[column_list]
    df.columns = ["smiles", "y", "id", "split", "source"]
    df["y"] = df["y"].clip(lower=-8, upper=-4)
    # print(df.head)

    grouped = df.groupby("split")
    for group_name, group_df in grouped:
        group_df.to_csv(f"temp_{group_name}.csv", index=False)

    # !!! make sure use_data_saving is false
    X_train, y_train = load_data_from_pdb(
        "temp_train.csv", pdb_dir, remove_h, one_hot_formal_charge=True
    )

    X_val, y_val = load_data_from_pdb(
        "temp_valid.csv", pdb_dir, remove_h, one_hot_formal_charge=True
    )

    X_test, y_test = load_data_from_pdb(
        "temp_test.csv", pdb_dir, remove_h, one_hot_formal_charge=True
    )

    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train)

    X_val = pd.DataFrame(X_val)
    y_val = pd.Series(y_val)

    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test)

    X_train.to_pickle(
        f"./pkl/pept_with_pdb_only/X_train_pdb_woH_{remove_h}_{split_seed}.pkl"
    )
    y_train.to_pickle(
        f"./pkl/pept_with_pdb_only/y_train_pdb_woH_{remove_h}_{split_seed}.pkl"
    )

    X_val.to_pickle(
        f"./pkl/pept_with_pdb_only/X_val_pdb_woH_{remove_h}_{split_seed}.pkl"
    )
    y_val.to_pickle(
        f"./pkl/pept_with_pdb_only/y_val_pdb_woH_{remove_h}_{split_seed}.pkl"
    )

    X_test.to_pickle(
        f"./pkl/pept_with_pdb_only/X_test_pdb_woH_{remove_h}_{split_seed}.pkl"
    )
    y_test.to_pickle(
        f"./pkl/pept_with_pdb_only/y_test_pdb_woH_{remove_h}_{split_seed}.pkl"
    )


if __name__ == "__main__":
    for i in range(0, 10):
        data_processor_from_pdb(
            "Random_Split_With_PDB.csv",
            pdb_dir="/home/liuwei/GitHub/Data/Hexene",
            remove_h=True,
            split_seed=i,
        )
