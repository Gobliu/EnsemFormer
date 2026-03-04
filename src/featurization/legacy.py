"""Legacy single-conformer dataset classes.

Used by single-conformer baselines. Kept for backward compatibility.
"""

from torch.utils.data import Dataset


class Molecule:
    """Single-conformer molecule data container."""

    def __init__(self, x, y, index):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.y = y
        self.index = index


class MolDataset(Dataset):
    """PyTorch Dataset wrapping a list of Molecule objects."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MolDataset(self.data_list[key])
        return self.data_list[key]
