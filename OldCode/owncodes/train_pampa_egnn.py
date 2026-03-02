import argparse
import json

import torch
from torch import nn, optim

import egnn.utils as utils
from egnn.qm9 import utils as qm9_utils
from egnn.qm9 import dataset
from egnn.qm9.models import EGNN

parser = argparse.ArgumentParser(description="PAMPA Prediction")
parser.add_argument("--exp_name", type=str, default="exp_1", help="experiment_name")
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs", type=int, default=1000, help="number of epochs to train (default: 10)"
)

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument(
    "--log_interval",
    type=int,
    default=20,
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--test_interval",
    type=int,
    default=1,
    help="how many epochs to wait before logging test",
)
parser.add_argument(
    "--outf", type=str, default="egnn/qm9/logs", help="folder to output vae"
)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--nf", type=int, default=128, help="hidden rate")
parser.add_argument(
    "--attention", type=int, default=1, help="attention in the ae model"
)
parser.add_argument(
    "--n_layers", type=int, default=7, help="number of layers for the autoencoder"
)
parser.add_argument(
    "--property",
    type=str,
    default="pampa",
    help="label to predict: pampa | To Be Added...",
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="number of workers for the dataloader"
)
parser.add_argument(
    "--charge_power",
    type=int,
    default=2,
    help="maximum power to take into one-hot features",
)
parser.add_argument(
    "--dataset_paper", type=str, default="cormorant", help="cormorant, lie_conv"
)
parser.add_argument("--node_attr", type=int, default=0, help="node_attr or not")
parser.add_argument("--weight_decay", type=float, default=1e-16, help="weight decay")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

utils.makedir(args.outf)
utils.makedir(args.outf + "/" + args.exp_name)

dataloaders, charge_scale = dataset.retrieve_dataloaders(
    args.batch_size, args.num_workers
)  # TODO check charge scale
# compute mean and mean absolute deviation
mean, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)

model = EGNN(
    in_node_nf=15,
    in_edge_nf=0,
    hidden_nf=args.nf,
    n_layers=args.n_layers,
    coords_weight=1.0,
    attention=args.attention,
    node_attr=args.node_attr,
).to(device)

print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()


def train(epoch, loader, partition="train"):
    res = {"loss": 0, "counter": 0, "loss_arr": []}
    for i, data in enumerate(loader):
        if partition == "train":
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        batch_size, n_nodes, _ = data["positions"].size()
        atom_positions = (
            data["positions"].view(batch_size * n_nodes, -1).to(device, dtype)
        )
        atom_mask = data["atom_mask"].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data["edge_mask"].to(device, dtype)
        one_hot = data["one_hot"].to(device, dtype)
        charges = data["charges"].to(device, dtype)
        nodes = qm9_utils.preprocess_input(
            one_hot, charges, args.charge_power, charge_scale, device
        )  # TODO

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[args.property].to(device, dtype)

        pred = model(
            h0=nodes,
            x=atom_positions,
            edges=edges,
            edge_attr=None,
            node_mask=atom_mask,
            edge_mask=edge_mask,
            n_nodes=n_nodes,
        )

        if partition == "train":
            loss = loss_l1(pred, (label - mean) / mad)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        else:
            loss = loss_l1(mad * pred + mean, label)

        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size
        res["loss_arr"].append(loss.item())

        prefix = ""
        if partition != "train":
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            print(
                prefix
                + "Epoch %d \t Iteration %d \t loss %.4f"
                % (epoch, i, sum(res["loss_arr"][-10:]) / len(res["loss_arr"][-10:]))
            )
    return res["loss"] / res["counter"]


if __name__ == "__main__":
    res = {
        "epochs": [],
        "losses": [],
        "best_val": 1e10,
        "best_test": 1e10,
        "best_epoch": 0,
    }

    for epoch in range(0, args.epochs):
        torch.set_float32_matmul_precision("high")
        train(epoch, dataloaders["train"], partition="train")
        if epoch % args.test_interval == 0:
            val_loss = train(epoch, dataloaders["valid"], partition="valid")
            test_loss = train(epoch, dataloaders["test"], partition="test")
            res["epochs"].append(epoch)
            res["losses"].append(test_loss)

            if val_loss < res["best_val"]:
                res["best_val"] = val_loss
                res["best_test"] = test_loss
                res["best_epoch"] = epoch
            print(
                "Val loss: %.4f \t test loss: %.4f \t epoch %d"
                % (val_loss, test_loss, epoch)
            )
            print(
                "Best: val loss: %.4f \t test loss: %.4f \t epoch %d"
                % (res["best_val"], res["best_test"], res["best_epoch"])
            )

        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losses.json", "w") as outfile:
            outfile.write(json_object)
