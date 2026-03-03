# Adapted from CPMP.data.pampa_uff_ig_true

import pandas as pd

from cpmp.featurization.data_utils import load_data_from_df


def data_processor_from_smiles(ip_csv_path, ff, ig, split_seed):
    column_list = [
        "SMILES",
        "PAMPA",
        "Original_Name_in_Source_Literature",
        f"split{split_seed}",
    ]
    print(column_list)
    df = pd.read_csv(ip_csv_path, low_memory=False)[column_list]
    df.columns = ["smiles", "y", "name", "split"]
    df["y"] = df["y"].clip(lower=-8, upper=-4)
    # print(df.head)

    grouped = df.groupby("split")
    for group_name, group_df in grouped:
        group_df.to_csv(f"temp_{group_name}.csv", index=False)

    # !!! make sure use_data_saving is false
    X_train, y_train = load_data_from_df(
        "temp_train.csv",
        ff=ff,
        ignoreInterfragInteractions=ig,
        one_hot_formal_charge=True,
        use_data_saving=False,
    )

    X_val, y_val = load_data_from_df(
        "temp_valid.csv",
        ff=ff,
        ignoreInterfragInteractions=ig,
        one_hot_formal_charge=True,
        use_data_saving=False,
    )

    X_test, y_test = load_data_from_df(
        "temp_test.csv",
        ff=ff,
        ignoreInterfragInteractions=ig,
        one_hot_formal_charge=True,
        use_data_saving=False,
    )

    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train)

    X_val = pd.DataFrame(X_val)
    y_val = pd.Series(y_val)

    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test)

    X_train.to_pickle(f"./pkl/pept_with_pdb_only/X_train_{ff}_{ig}_{split_seed}.pkl")
    y_train.to_pickle(f"./pkl/pept_with_pdb_only/y_train_{ff}_{ig}_{split_seed}.pkl")

    X_val.to_pickle(f"./pkl/pept_with_pdb_only/X_val_{ff}_{ig}_{split_seed}.pkl")
    y_val.to_pickle(f"./pkl/pept_with_pdb_only/y_val_{ff}_{ig}_{split_seed}.pkl")

    X_test.to_pickle(f"./pkl/pept_with_pdb_only/X_test_{ff}_{ig}_{split_seed}.pkl")
    y_test.to_pickle(f"./pkl/pept_with_pdb_only/y_test_{ff}_{ig}_{split_seed}.pkl")


if __name__ == "__main__":
    for i in range(0, 10):
        data_processor_from_smiles(
            "Random_Split_With_PDB.csv", ff="uff", ig=True, split_seed=i
        )
        data_processor_from_smiles(
            "Random_Split_With_PDB.csv", ff="uff", ig=False, split_seed=i
        )
        data_processor_from_smiles(
            "Random_Split_With_PDB.csv", ff="mmff", ig=True, split_seed=i
        )
        data_processor_from_smiles(
            "Random_Split_With_PDB.csv", ff="mmff", ig=False, split_seed=i
        )
