import argparse
from typing import Union
import random
import torch
from model.clean_egnn import get_edges_batch
import numpy as np


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def collate_fn(batch):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
    batch_size, n_nodes, _ = batch['coords'].shape

    included_species = torch.unique(batch['z'], sorted=True)
    if included_species[0] == 0:
        included_species = included_species[1:]
    one_hot = batch['z'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
    charges = batch['z'].float()

    charge_scale = max(included_species)
    nodes = preprocess_input(one_hot, charges, charge_power=2, charge_scale=charge_scale, device='cpu')

    coords = batch['coords'].view(batch_size * n_nodes, -1).float()
    nodes = nodes.view(batch_size * n_nodes, -1).float()

    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    batch['nodes'] = nodes
    batch['coords'] = coords
    batch['edges'] = edges
    batch['edge_attr'] = edge_attr
    batch['energies'] = batch['energies'].view(-1)
    batch['n_nodes'] = n_nodes

    batch['forces'] = batch['forces'].view(batch_size, n_nodes, -1).float()

    batch['energy_meann'] = batch['energy_meann'].float()
    batch['energy_mad'] = batch['energy_mad'].float()
    batch['force_meann'] = batch['force_meann'].float()
    batch['force_mad'] = batch['force_mad'].float()

    batch['batch_size'] = batch_size

    return batch


class EnergyDataset:

    def __init__(self, file_path: str, train: bool = True, seed: int = 123):
        """
        Dataset for a molecule
        :param file_path: path to the *.npz file
        """
        random.seed(seed)
        data = np.load(file_path)
        num_samples = data['F'].shape[0]
        all_idx = [i for i in range(num_samples)]
        random.shuffle(all_idx)

        train_start_idx = 0
        train_end_idx = 50000

        test_start_idx = 50000
        test_end_idx = 60000

        if train:
            start_idx = train_start_idx
            end_idx = train_end_idx
        else:
            start_idx = test_start_idx
            end_idx = test_end_idx

        self.energies = data['E'][all_idx[start_idx: end_idx]]
        self.forces = data['F'][all_idx[start_idx: end_idx]]

        self.energy_meann = np.mean(data['E'][all_idx[train_start_idx: train_end_idx]])
        ma = np.abs(data['E'][all_idx[train_start_idx: train_end_idx]] - self.energy_meann)
        self.energy_mad = np.mean(ma)

        self.force_meann = np.mean(data['F'][all_idx[train_start_idx: train_end_idx]])
        ma = np.abs(data['F'][all_idx[train_start_idx: train_end_idx]] - self.force_meann)
        self.force_mad = np.mean(ma)

        self.coords = data['R'][all_idx[start_idx: end_idx]]
        self.z = data['z'].astype(np.int64)

    def __getitem__(self, item):
        coords, z, energies, forces = self.coords[item], self.z, self.energies[item], self.forces[item]
        return {'coords': torch.tensor(coords, dtype=torch.float32),
                'z': torch.tensor(z),
                'energies': torch.tensor(energies),
                'forces': torch.tensor(forces),
                'energy_meann': self.energy_meann,
                'energy_mad': self.energy_mad,
                'force_meann': self.force_meann,
                'force_mad': self.force_mad}

    def __len__(self):
        return self.coords.shape[0]


def get_dataset(opts: argparse.Namespace, train: bool = True, seed: int = 123) -> EnergyDataset:
    """
    Get the appropriate dataset

    :param opts: the options object
    :param train: if it is training or validation
    :return: the dataset object
    """
    return EnergyDataset(opts.dataset_path, train=train, seed=seed)


if __name__ == '__main__':
    pass
