# Adapted from CPMP.data.pampa_uff_ig_true

import pandas as pd

from LoadDataFromPDB import load_data_from_pdb


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
