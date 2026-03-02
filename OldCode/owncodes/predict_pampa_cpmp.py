import os
import sys
import numpy as np
import pandas as pd
import torch
import itertools
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from cpmp.featurization.data_utils import construct_loader
from cpmp.model.transformer import make_model


def predict(model, data_loader_train, criterion, device):
    outputs = []
    targets = []
    model = model.eval()
    sample_size = 0
    total_loss = 0
    for batch in data_loader_train:
        adjacency_matrix, node_features, distance_matrix, y = batch
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

        # output = 2 * output - 6

        outputs.extend(output.view(1, -1)[0].detach().cpu().numpy().tolist())
        sample_size += len(y)
    r2 = metrics.r2_score(targets, outputs)
    mse = metrics.mean_squared_error(outputs, targets)
    mae = metrics.mean_absolute_error(outputs, targets)
    return outputs, total_loss / sample_size, r2, mse, mae


# one_hot_formal_charge=True
def main(data_dict, split_seed):
    dataset = "pampa_with_pdb"
    outdir = f"../../SavedModel/train_{dataset}_CPMP"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    h_list = [64]
    N_list = [6]
    N_dense_list = [2]
    slope_list = [0.16]

    parameter_combinations = list(
        itertools.product(h_list, N_list, N_dense_list, slope_list)
    )
    for h, N, N_dense, slope in parameter_combinations:
        pkl_dir = "./DataProcessor/pkl/pept_with_pdb_only"

        if "pdb" in data_dict:
            X_test = pd.read_pickle(
                f"{pkl_dir}/X_test_pdb_wH_{data_dict['wh']}_{split_seed}.pkl"
            ).values.tolist()
            y_test = pd.read_pickle(
                f"{pkl_dir}/y_test_pdb_wH_{data_dict['wh']}_{split_seed}.pkl"
            ).values.tolist()
        else:
            ff, ig = data_dict["ff"], data_dict["ig"]
            X_test = pd.read_pickle(
                f"{pkl_dir}/X_test_{ff}_{ig}_{split_seed}.pkl"
            ).values.tolist()
            y_test = pd.read_pickle(
                f"{pkl_dir}/y_test_{ff}_{ig}_{split_seed}.pkl"
            ).values.tolist()

        d_atom = X_test[0][0].shape[1]
        d_model = 64
        h = h
        N = N
        N_dense = N_dense
        slope = slope
        drop = 0.1
        lambda_attention = 0.1
        lambda_distance = 0.6
        aggregation = "dummy_node"
        gpu = "cuda:0"
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

        if "pdb" in data_dict:
            best_model_saver = f"{outdir}/seed{split_seed}_pdb_wH_{data_dict['wh']}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.best_weight.pth"
            LOG_FILE = f"{outdir}/seed{split_seed}_pdb_wH_{data_dict['wh']}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.predict.csv"
        else:
            best_model_saver = f"{outdir}/seed{split_seed}_{ff}_ig_{ig}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.best_weight.pth"
            LOG_FILE = f"{outdir}/seed{split_seed}_{ff}_ig_{ig}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.predict.csv"

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = torch.device(gpu)
            pretrained_state_dict = torch.load(
                best_model_saver, weights_only=True, map_location=torch.device(gpu)
            )
        else:
            device = torch.device("cpu")
            pretrained_state_dict = torch.load(
                best_model_saver, weights_only=True, map_location=torch.device("cpu")
            )
        model = make_model(**model_params)
        model_state_dict = model.state_dict()
        for name, param in pretrained_state_dict.items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

        model = model.to(device)
        data_loader_test = construct_loader(X_test, y_test, batch_size, shuffle=False)
        criterion = nn.HuberLoss()
        y_predict, test_loss, r2, mse, mae = predict(
            model, data_loader_test, criterion, device
        )
        print(f"test_loss={test_loss:.5f}, r2={r2:.5f}, mse={mse:.5f}, mae={mae:.5f}")

        y = [item for sublist in y_test for item in sublist]
        df = pd.DataFrame({"y": y})
        df["predict"] = y_predict
        df.to_csv(LOG_FILE, mode="w", index=False, header=True)

        threshold = -6

        df["y_binary"] = (df["y"] > threshold).astype(int)

        # 计算 AUC
        auc_score = roc_auc_score(df["y_binary"], df["predict"])
        print(f"ROC-AUC: {auc_score}")

        # 计算 PRC
        precision, recall, _ = precision_recall_curve(df["y_binary"], df["predict"])
        prc_auc = auc(recall, precision)
        print(f"PR-AUC: {prc_auc}")

        # 计算精度
        df["predict_binary"] = (df["predict"] > threshold).astype(int)
        accuracy = accuracy_score(df["y_binary"], df["predict_binary"])
        print(f"Accuracy: {accuracy}")

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(df["y_binary"], df["predict_binary"])
        print("Confusion Matrix:")
        print(conf_matrix)

        # 打印分类报告
        class_report = classification_report(df["y_binary"], df["predict_binary"])
        print("Classification Report:")
        print(class_report)

        return r2, mse, mae, auc_score


if __name__ == "__main__":
    r2_list = []
    mse_list = []
    mae_list = []
    roc_auc_list = []
    for i in range(0, 10):
        # data_dict = {'ff': 'mmff', 'ig': False, 'wh': False}
        data_dict = {"pdb": None, "wh": False}
        r2, mse, mae, auc_score = main(data_dict, split_seed=i)
        r2_list.append(r2)
        mse_list.append(mse)
        mae_list.append(mae)
        roc_auc_list.append(auc_score)
    print("r2:", np.mean(r2_list), np.std(r2_list, ddof=1), r2_list)
    print("mse:", np.mean(mse_list), np.std(mse_list, ddof=1), mse_list)
    print("mae:", np.mean(mae_list), np.std(mae_list, ddof=1), mae_list)
    print("roc-auc:", np.mean(roc_auc_list), np.std(roc_auc_list, ddof=1), roc_auc_list)
