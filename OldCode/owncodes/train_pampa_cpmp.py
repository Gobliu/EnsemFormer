import os
import sys
import pandas as pd
import torch
import itertools
import torch.nn as nn
from sklearn import metrics
import torch.optim as optim


from cpmp.featurization.data_utils import construct_loader
from cpmp.model.transformer import CPMPGraphTransformer


def train(model, data_loader_train, criterion, lr, device):
    sample_size = 0
    total_loss = 0
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for batch in data_loader_train:
        adjacency_matrix, node_features, distance_matrix, y = batch

        # y = (y + 6) / 2               # tried normalization, no significant difference

        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        adjacency_matrix = adjacency_matrix.to(device)
        node_features = node_features.to(device)
        distance_matrix = distance_matrix.to(device)
        batch_mask = batch_mask.to(device)
        y = y.to(device)
        output = model(
            node_features, batch_mask, adjacency_matrix, distance_matrix, None
        )
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        sample_size += len(y)
    return total_loss / sample_size


def evaluate(model, data_loader_train, criterion, lr, device):
    outputs = []
    targets = []
    model = model.eval()
    sample_size = 0
    total_loss = 0
    for batch in data_loader_train:
        adjacency_matrix, node_features, distance_matrix, y = batch

        # y = (y + 6) / 2               # tried normalization, perform worse

        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        adjacency_matrix = adjacency_matrix.to(device)
        node_features = node_features.to(device)
        distance_matrix = distance_matrix.to(device)
        batch_mask = batch_mask.to(device)
        targets.extend(y.view(1, -1)[0].numpy().tolist())
        y = y.to(device)
        output = model(
            node_features, batch_mask, adjacency_matrix, distance_matrix, None
        )

        loss = criterion(output, y)
        total_loss += loss.item()
        outputs.extend(output.view(1, -1)[0].detach().cpu().numpy().tolist())
        sample_size += len(y)

    r2 = metrics.r2_score(targets, outputs)
    mse = metrics.mean_squared_error(outputs, targets)
    mae = metrics.mean_absolute_error(outputs, targets)
    return total_loss / sample_size, r2, mse, mae


def main(data_dict, split_seed):
    torch.cuda.empty_cache()
    torch.manual_seed(123 * split_seed**2)
    dataset = "pampa_with_pdb"
    outdir = f"results/train_{dataset}_CPMP"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    h_list = [64]  # original 64, change to 8
    N_list = [6]
    N_dense_list = [2]
    slope_list = [0.16]

    parameter_combinations = list(
        itertools.product(h_list, N_list, N_dense_list, slope_list)
    )
    for h, N, N_dense, slope in parameter_combinations:

        # for pdb generated data
        pkl_dir = "owncodes/DataProcessor/pkl"
        if "pdb" in data_dict:
            X_train = pd.read_pickle(
                f"{pkl_dir}/X_train_pdb_wH_{data_dict['wh']}_{split_seed}.pkl"
            ).values.tolist()
            X_val = pd.read_pickle(
                f"{pkl_dir}/X_val_pdb_wH_{data_dict['wh']}_{split_seed}.pkl"
            ).values.tolist()
            X_test = pd.read_pickle(
                f"{pkl_dir}/X_test_pdb_wH_{data_dict['wh']}_{split_seed}.pkl"
            ).values.tolist()
            y_train = pd.read_pickle(
                f"{pkl_dir}/y_train_pdb_wH_{data_dict['wh']}_{split_seed}.pkl"
            ).values.tolist()
            y_val = pd.read_pickle(
                f"{pkl_dir}/y_val_pdb_wH_{data_dict['wh']}_{split_seed}.pkl"
            ).values.tolist()
            y_test = pd.read_pickle(
                f"{pkl_dir}/y_test_pdb_wH_{data_dict['wh']}_{split_seed}.pkl"
            ).values.tolist()
        else:
            ff, ig = data_dict["ff"], data_dict["ig"]
            X_train = pd.read_pickle(
                f"{pkl_dir}/X_train_{ff}_{ig}_{split_seed}.pkl"
            ).values.tolist()
            X_val = pd.read_pickle(
                f"{pkl_dir}/X_val_{ff}_{ig}_{split_seed}.pkl"
            ).values.tolist()
            X_test = pd.read_pickle(  # noqa: F841
                f"{pkl_dir}/X_test_{ff}_{ig}_{split_seed}.pkl"
            ).values.tolist()
            y_train = pd.read_pickle(
                f"{pkl_dir}/y_train_{ff}_{ig}_{split_seed}.pkl"
            ).values.tolist()
            y_val = pd.read_pickle(
                f"{pkl_dir}/y_val_{ff}_{ig}_{split_seed}.pkl"
            ).values.tolist()
            y_test = pd.read_pickle(  # noqa: F841
                f"{pkl_dir}/y_test_{ff}_{ig}_{split_seed}.pkl"
            ).values.tolist()

        """
        After the hyperparameters are determined, we merge the training set and the validation set to train the model.
        """
        # X_train.extend(X_val)
        # y_train.extend(y_val)

        d_atom = X_train[0][0].shape[1]
        d_model = 64  # original 64,  test 128, 256, 512
        h = h
        N = N
        N_dense = N_dense
        slope = slope
        drop = 0.1  # test 0.1, 0.3
        lambda_attention = 0.1
        lambda_distance = 0.6
        aggregation = "dummy_node"
        gpu = "cuda:0"
        epochs = 7
        max_patience = 100
        model_params = {
            "d_atom": d_atom,
            "d_model": d_model,
            "N": N,
            "h": h,
            "N_dense": N_dense,
            "lambda_attention": lambda_attention,
            "lambda_distance": lambda_distance,
            "leaky_relu_slope": slope,
            "dense_output_nonlinearity": "relu",
            "distance_matrix_kernel": "exp",
            "dropout": drop,
            "aggregation_type": aggregation,
        }
        batch_size = 64
        lr = 1e-3  # original 1e-3, tested 1e-4, 0.5e-4

        # # for pdb generated data
        if "pdb" in data_dict:
            best_model_saver = f"{outdir}/seed{split_seed}_pdb_wH_{data_dict['wh']}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.best_weight.pth"
            LOG_FILE = f"{outdir}/seed{split_seed}_pdb_wH_{data_dict['wh']}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.log.csv"
        else:
            best_model_saver = f"{outdir}/seed{split_seed}_{ff}_ig_{ig}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.best_weight.pth"
            LOG_FILE = f"{outdir}/seed{split_seed}_{ff}_ig_{ig}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.log.csv"

        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

        best_loss = sys.maxsize
        use_cuda = torch.cuda.is_available()
        device = torch.device(gpu if use_cuda else "cpu")
        model = CPMPGraphTransformer(**model_params)
        model = model.to(device)
        patience = 0
        for epoch in range(1, epochs + 1):
            data_loader_train = construct_loader(X_train, y_train, batch_size)
            # data_loader_test = construct_loader(X_test, y_test, batch_size)
            data_loader_valid = construct_loader(X_val, y_val, batch_size)
            criterion = nn.MSELoss(
                reduction="sum"
            )  # original is MSELoss, tested huber, smoothL1, L1
            train_loss = train(model, data_loader_train, criterion, lr, device)
            valid_loss, r2, mse, mae = evaluate(
                model, data_loader_valid, criterion, lr, device
            )

            print(
                f"epoch={epoch}, train_loss={train_loss:.5f}, valid_loss={valid_loss:.5f},"
                f" r2={r2:.5f}, mse={mse:.5f}, mae={mae:.5f}, best_loss={best_loss:.5f}, patience={patience}"
            )

            if not best_loss or best_loss > valid_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), best_model_saver)
                patience = 0
            else:
                patience += 1

            results = pd.DataFrame(
                {
                    "epoch": [epoch],
                    "train_loss": [train_loss],
                    "test_loss": [valid_loss],
                    "r2": [r2],
                    "mse": [mse],
                    "mae": [mae],
                    "best_loss": [best_loss],
                    "patience": [patience],
                }
            )
            # print(LOG_FILE, results)
            results.to_csv(
                LOG_FILE,
                mode="a",
                index=False,
                header=False if os.path.isfile(LOG_FILE) else True,
            )

            if patience > max_patience:
                break


if __name__ == "__main__":
    # for i in range(
    #     0, 10
    # ):  # range(1, 11) for 2d benchmark data, range(0, 10) for 3d benchmark data
    # data_dict = {"ff": "uff", "ig": True, "wh": False}
    # main(data_dict, split_seed=i)
    # data_dict = {"ff": "uff", "ig": False, "wh": False}
    # main(data_dict, split_seed=i)
    # data_dict = {"ff": "mmff", "ig": True, "wh": False}
    # main(data_dict, split_seed=i)
    data_dict = {"ff": "mmff", "ig": False, "wh": False}
    main(data_dict, split_seed=0)
    # data_dict = {"pdb": None, "wh": False}
    # main(data_dict, split_seed=i)
