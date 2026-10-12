"""Molecule -> heavy-atom graph featuriser, identical to QGNN_01_Subset_GraphDataset.ipynb.

Node features (23): element C/N/O/F, degree 0-4, attached H 0-4, hybridisation sp/sp2/sp3/other,
aromatic, in ring, formal charge -1/0/+1.  Edge features (6): single/double/triple/aromatic,
conjugated, in ring.  Edges are stored in both directions.
"""
import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset

RDLogger.DisableLog("rdApp.*")

ELEMENTS   = ["C", "N", "O", "F"]
DEGREES    = [0, 1, 2, 3, 4]
NUM_HS     = [0, 1, 2, 3, 4]
HYBRIDS    = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
CHARGES    = [-1, 0, 1]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def one_hot(value, choices, allow_other=False):
    v = [float(value == c) for c in choices]
    if allow_other:
        v.append(float(value not in choices))
    elif sum(v) == 0:
        raise ValueError(f"Unexpected value {value!r}; expected one of {choices}")
    return v


def atom_features(a):
    return (one_hot(a.GetSymbol(), ELEMENTS) + one_hot(a.GetDegree(), DEGREES)
            + one_hot(a.GetTotalNumHs(), NUM_HS) + one_hot(a.GetHybridization(), HYBRIDS, allow_other=True)
            + [float(a.GetIsAromatic()), float(a.IsInRing())] + one_hot(a.GetFormalCharge(), CHARGES))


def bond_features(b):
    return one_hot(b.GetBondType(), BOND_TYPES) + [float(b.GetIsConjugated()), float(b.IsInRing())]


def smiles_to_data(smiles, y_scaled, y_eV, row):
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        src += [i, j]; dst += [j, i]; ea += [f, f]
    return Data(x=x,
                edge_index=torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
                edge_attr=torch.tensor(ea, dtype=torch.float32).reshape(-1, 6),
                y=torch.tensor([[y_scaled]], dtype=torch.float32),
                y_eV=torch.tensor([[y_eV]], dtype=torch.float32),
                row=torch.tensor([row]))


class GraphSet(InMemoryDataset):
    """Wraps a collated (data, slices) pair so that ds[i] / ds[index_tensor] work."""
    def __init__(self, data, slices):
        super().__init__(root=None)
        self.data, self.slices = data, slices

    @classmethod
    def from_file(cls, path):
        blob = torch.load(path, weights_only=False)
        ds = cls(blob["data"], blob["slices"])
        ds.extra = {k: v for k, v in blob.items() if k not in ("data", "slices")}
        return ds
