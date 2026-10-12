# GNN vs Quantum-Inspired GNN vs Quantum GNN on QM9: Project Guide

*A walkthrough of everything built so far, written for someone picking up the **classical GNN** part.*

---

## 1. What the project is

We predict one property of small organic molecules, the **HOMO–LUMO gap (in eV)**, with three families of models:

| Model | Runs on | Idea |
|---|---|---|
| **Classical GNN** (your part) | CPU/GPU | Atoms are nodes and bonds are edges. Message passing (GCN / GIN / MPNN) learns a vector for each molecule, and a regressor predicts the gap. |
| **Quantum-inspired GNN (QI-GNN)** | CPU/GPU | A classical GNN with added physics: edge embeddings built from 3D interatomic distances and directions. |
| **Quantum GNN** | Simulated qubits | One qubit per heavy atom. A variational circuit entangles qubits along the bonds, and qubit measurements give the prediction. |

**Fair comparison rule:** every model uses the same molecules, the same splits, the same target and the same metric (**MAE in eV**). We also report parameter count and training time.

The classical GNN matters in two ways. It is the **baseline** the other two are judged against. It also **validates the data pipeline**: if a standard GIN on these graphs gets close to published numbers, the graphs are built correctly. Any later problem then lies in the new models, not in the data.

---

## 2. The dataset: QM9

- About 134k small organic molecules, each with **at most 9 heavy atoms** (C, N, O, F) plus hydrogens.
- Every molecule has properties computed with DFT (B3LYP/6-31G(2df,p)).
- Source file: `qm9.csv` from MoleculeNet/DeepChem. It holds SMILES strings plus 12 properties, but **no 3D coordinates**.

| Property group | Columns | Behaviour |
|---|---|---|
| Electronic | homo, lumo, gap, mu | Intensive (does not grow with size) |
| Size / shape | alpha, r2, A, B, C | Mostly extensive |
| Thermochemical | u0, u298, h298, g298, zpve, cv | Extensive (grows with size) |

### Why the HOMO–LUMO gap?
- **Intensive:** a model cannot do well just by counting atoms.
- **Well-behaved:** almost symmetric (skew ≈ 0), mean 6.86 eV, std 1.28 eV, very few outliers.
- **Meaningful:** it governs reactivity, colour and electronic behaviour.
- **Benchmarkable:** published papers report gap MAE, so we have targets to compare with.

### ⚠️ Leakage rule
`gap = lumo − homo` exactly, so **homo and lumo must never be inputs**. No other DFT property is used as an input either. Inputs come **only from the molecule's structure** (the SMILES).

---

## 3. Step 1 (done): EDA and preprocessing (`QM9_EDA_Preprocessing.ipynb`)

### What the EDA found
- **The data is clean:** no missing values and no invalid SMILES.
- **83 duplicate SMILES** were removed using RDKit canonical SMILES.
- **3,054 "uncharacterised" molecules** were removed. These failed QM9's own geometry consistency check and are a known issue in the dataset.
- Result: **133,885 → 130,745 molecules.**
- **The chemistry is imbalanced:**
  - C is in ≈100% of molecules, O in 85%, N in 61%, but **F in only 1.45%**. Expect larger errors on fluorine molecules.
  - 83% of molecules have exactly 9 heavy atoms, 67% have 1–2 rings, and 17% have an aromatic ring.

### What preprocessing did
1. Converted HOMO, LUMO and gap from Hartree to **eV**.
2. Built two classical feature sets from SMILES, used for non-graph baselines:
   - 165 cleaned **RDKit descriptors** (after dropping near-constant and highly correlated ones),
   - 1024-bit **Morgan fingerprints**.
3. **Split before fitting anything:** 80 / 10 / 10 random split (seed 42) → **104,595 train / 13,075 val / 13,075 test**.
4. **Scaled the target** with a StandardScaler fit on train only: `y_scaled = (gap_eV − 6.8583) / 1.2837`.
5. For a simple quantum baseline, compressed the descriptors to **8 PCA components** and rescaled them to angles in [0, π] (one angle per qubit).
6. Saved everything to `processed/`:
   - `qm9_clean.csv`: the 130,745 cleaned molecules with all properties.
   - `qm9_gap_eV_processed.npz`: descriptors, fingerprints, PCA angles, scaled targets, and the split indices `idx_tr / idx_va / idx_te` (row positions in `qm9_clean.csv`).
   - `qm9_gap_eV_transforms.joblib`: the fitted scalers and PCA.

---

## 4. Step 2 (done): subset and graph dataset (`QGNN_01_Subset_GraphDataset.ipynb`)

This notebook turns molecules into graphs. **This is the data you will train on.**

### 4.1 Why a subset?
The quantum model has to be **simulated**: every molecule passes through a qubit circuit at every training step. That is far too slow for 104k molecules. So we use a **14,000-molecule subset**. For fairness, **the classical GNN must also be trained on this subset**, and separately on the full data as a reference.

### 4.2 How the subset was chosen
- **10,000 train / 2,000 val / 2,000 test**, each drawn **from inside** the original train / val / test sets, so no molecule changes side.
- Sampling is **stratified** on *gap decile × contains-fluorine*. The subset keeps the gap distribution, and the rare F molecules still appear in test.
- Because the subset test set is part of the full test set, a GNN trained on all 104k molecules can be scored on **exactly the same 2,000 test molecules**.

The subset matches the full data closely:

| | Full train | Subset train | Subset val | Subset test |
|---|---|---|---|---|
| Molecules | 104,595 | 10,000 | 2,000 | 2,000 |
| Gap mean (eV) | 6.858 | 6.861 | 6.865 | 6.858 |
| Gap std (eV) | 1.284 | 1.285 | 1.282 | 1.274 |
| % with F | 1.47 | 1.46 | 1.45 | 1.35 |
| % with 9 heavy atoms | 83.4 | 83.3 | 83.6 | 84.2 |
| % with aromatic ring | 16.9 | 16.5 | 17.8 | 16.6 |

### 4.3 How a molecule becomes a graph
We use a **heavy-atom graph**:
- **Nodes** are the C, N, O and F atoms.
- **Hydrogens are not nodes.** The number of attached H atoms is a node feature instead.
- Every molecule therefore has **≤ 9 nodes**. That is what lets the quantum model use one qubit per atom. The classical GNN uses the same graphs so the comparison stays fair.
- On average a graph has 8.8 nodes and 9.4 bonds.

**Node features (23 per atom):**

| Feature | Encoding | Size |
|---|---|---|
| Element | one-hot C, N, O, F | 4 |
| Heavy-atom degree | one-hot 0–4 | 5 |
| Attached hydrogens | one-hot 0–4 | 5 |
| Hybridisation | one-hot sp, sp², sp³, other | 4 |
| Aromatic | 0/1 | 1 |
| In ring | 0/1 | 1 |
| Formal charge | one-hot −1, 0, +1 (QM9 has a few zwitterions like `[NH3+]` / `[O-]`) | 3 |

**Edge features (6 per bond):** bond type one-hot (single, double, triple, aromatic) plus conjugated and in-ring flags.

Edges are stored **in both directions** (i→j and j→i), as PyG message passing expects.

Example: ethane `CC` has 2 nodes, each an sp³ carbon with degree 1 and 3 H, joined by one single bond (2 directed edges).

### 4.4 Two splits over the same 14,000 molecules
- **Random split:** the inherited one. This is the standard QM9 benchmark setting.
- **Scaffold split:** molecules that share a Bemis–Murcko scaffold (the core ring system) are kept in the same split. Test molecules have ring systems **never seen in training**, so this tests generalisation to new chemistry. **0 scaffolds are shared between scaffold-train and scaffold-test.**

Why both? Published work (Therrien et al., below) shows that QM9 random-split scores **overstate** accuracy on truly new molecules: their GCN gets 0.12 eV on the QM9 test set but 0.96 eV on new molecules. Reporting both splits makes the comparison honest.

### 4.5 Saved files (`processed/graphs/`)

| File | Contents | Who uses it |
|---|---|---|
| `qm9_gap_subset_graphs.pt` | 14,000 PyG graphs (collated) + both splits + metadata | **Classical GNN**, QI-GNN |
| `qm9_gap_subset_qgnn.npz` | Same graphs as fixed-size arrays: atom features `[N, 9, 23]`, bond-type matrix `[N, 9, 9]` (0 = none, 1–4 = bond type), mask `[N, 9]` | Quantum GNN |
| `qm9_gap_subset_index.csv` | One row per molecule: row in `qm9_clean.csv`, mol_id, SMILES, scaffold, split labels | Joining predictions back to molecules |
| `qm9_gap_subset_meta.json` | Feature names, target mean/std, sizes, seed | Reproducibility |

Each PyG `Data` object contains:
- `x` `[n_atoms, 23]`, `edge_index` `[2, 2·n_bonds]`, `edge_attr` `[2·n_bonds, 6]`
- `y`: **scaled** target (train on this)
- `y_eV`: target in eV (for checking)
- `row`, `sub_idx`, `smiles`, `mol_id`: for tracing molecules

---

## 5. Your part: the classical GNN

> **Status:** this is now implemented and run in `QGNN_04_ClassicalGNN.ipynb` (GINE/GCN/GAT, subset and full data, random and scaffold splits). The code below is the minimal version of it, useful for reading or extending. Results are in section 8.

### 5.1 Loading the data

```python
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

class QM9Subset(InMemoryDataset):
    def __init__(self, path):
        super().__init__(root=None)
        blob = torch.load(path, weights_only=False)   # file holds PyG objects, not only tensors
        self.data, self.slices = blob["data"], blob["slices"]
        self.splits, self.meta = blob["splits"], blob["meta"]

ds = QM9Subset("processed/graphs/qm9_gap_subset_graphs.pt")
SPLIT = "random"                       # or "scaffold"
train_loader = DataLoader(ds[ds.splits[SPLIT]["train"]], batch_size=64, shuffle=True)
val_loader   = DataLoader(ds[ds.splits[SPLIT]["val"]],   batch_size=256)
test_loader  = DataLoader(ds[ds.splits[SPLIT]["test"]],  batch_size=256)
Y_STD = ds.meta["y_std_eV"]            # 1.2837 eV, to convert errors back to eV
```

Requirements: `torch`, `torch_geometric`, `rdkit` (RDKit is only needed to rebuild the graphs). The `.pt` file was saved with PyTorch 2.14 / PyG 2.8. If it will not load on your versions, re-run `QGNN_01_Subset_GraphDataset.ipynb`, which takes about 25 seconds.

### 5.2 Starter model: GINE

**GIN** (Graph Isomorphism Network) is one of the most expressive standard message-passing GNNs. **GINEConv** is the variant that also uses **edge features**, which matters because bond type carries a lot of chemistry.

```python
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

class GIN(nn.Module):
    def __init__(self, node_dim=23, edge_dim=6, hidden=128, layers=4):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                     edge_dim=edge_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, d):
        h = self.embed(d.x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.relu(norm(conv(h, d.edge_index, d.edge_attr)))   # residual connection
        return self.head(global_mean_pool(h, d.batch))                      # one vector per molecule

model = GIN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(loader):
    model.eval(); errs = []
    with torch.no_grad():
        for d in loader:
            errs.append((model(d) - d.y).abs().squeeze(1) * Y_STD)   # MAE back in eV
    return torch.cat(errs).mean().item()

for epoch in range(1, 201):
    model.train()
    for d in train_loader:
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(d), d.y)
        loss.backward(); opt.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch} | val MAE {evaluate(val_loader):.3f} eV")
```

This is a starting point, not a tuned model. A quick smoke test of this architecture (~156k parameters) reached a val MAE of about 0.32 eV after **1 epoch** on the 10k subset, so the pipeline works end to end. It has not been trained to convergence yet.

**How the model works:**
1. **Embed:** each atom's 23 features → a 128-dim vector.
2. **Message passing (×4):** each atom gathers its neighbours' vectors, combined with the bond features, and updates its own vector. After 4 layers, each atom "knows" its 4-bond neighbourhood, which covers most of a ≤9-atom molecule.
3. **Readout:** average the atom vectors into one molecule vector. Mean pooling suits an **intensive** target like the gap. Sum pooling is a reasonable alternative to try.
4. **Head:** an MLP turns that vector into one number (the scaled gap).

### 5.3 Suggested experiments

| # | Experiment | Why |
|---|---|---|
| 1 | GINE on the **full** 104k train set, scored on the 2,000-molecule subset test set | Checks the pipeline against the literature (~0.12 eV for a 2D GCN) |
| 2 | GINE on the **10k subset**, random split | The fair baseline for the quantum model |
| 3 | Same as #2 on the **scaffold split** | Generalisation to unseen chemistry |
| 4 | Swap the layer: GCN, GAT, MPNN (`NNConv`) | The Li et al. paper suggests GAT/GIN; good for the comparison table |
| 5 | **Small GNN** with a parameter count close to the quantum model (tens to a few hundred parameters) | The quantum model will have very few parameters; this checks whether a classical model of the same size does as well |
| 6 | Per-element error (molecules with vs without F) | Expect larger errors on the rare F molecules |

**Practical tips:**
- Use early stopping on val MAE and a learning-rate scheduler (`ReduceLROnPlateau`).
- Run 3 seeds per setting and report mean ± std.
- Always report MAE **in eV**: multiply the scaled error by `Y_STD`.
- Log parameter count and wall-clock training time; both go in the final comparison table.
- Never touch the test set for tuning.

---

## 6. Reference numbers from the literature

| Paper | Model | Gap MAE | Note |
|---|---|---|---|
| Therrien, Sargent & Voznyy, *Nature Communications* 16:4301 (2025) | GCN (2D graph only) | **0.12 eV** | The classical number to aim for. On 1,617 new molecules: 0.96 eV |
| Same paper | PaiNN (uses 3D) | 0.048 eV | 3D geometry helps a lot |
| Chanana, Springer Proc. Physics 369 (IC-AMSI 2026) | QIGNN (distance + direction edge embeddings) | 0.1186 eV | Reference for our QI-GNN. "Quantum-inspired" here means physics-informed geometry, not qubits |
| Li et al., *Chemical Papers* 80 (2026) | MPNN, GAT, GIN, PhysNet | Dipole moment target | PhysNet most accurate; GAT degrades least on new molecules. A template for fair multi-model comparison |

Our subset models train on ~10× less data than these papers, so expect higher errors on the 10k subset. That is why experiment #1 (full data) matters.

---

## 7. Project files at a glance

```
qauntum/
├── qm9.csv                              raw QM9 (MoleculeNet)
├── QM9_EDA_Preprocessing.ipynb          step 1: EDA, cleaning, features, split, scaling
├── QGNN_01_Subset_GraphDataset.ipynb    step 2: subset + graphs + scaffold split
├── QGNN_02_QuantumGNN.ipynb             step 3: quantum circuit built and tested (vs PennyLane)
├── QGNN_03_QuantumGNN_Training.ipynb    step 4: quantum training, ablations, seeds, scaffold
├── QGNN_04_ClassicalGNN.ipynb           step 5: GINE / GCN / GAT, subset + full data
├── QGNN_05_Comparison.ipynb             step 6: one table + plots across all models
├── qgnn_lib.py                          quantum model, fast simulator, classical twin, training loop
├── qm9_graphs.py                        molecule -> graph featuriser (same as notebook 01)
├── results/                             *_results.csv written by notebooks 03 and 04
├── GNN vs Quantum GNN on QM9.pdf        project proposal slides
├── processed/
│   ├── qm9_clean.csv                    130,745 cleaned molecules
│   ├── qm9_gap_eV_processed.npz         descriptors, fingerprints, PCA angles, full split
│   ├── qm9_gap_eV_transforms.joblib     fitted scalers + PCA
│   └── graphs/
│       ├── qm9_gap_subset_graphs.pt     ← classical GNN input
│       ├── qm9_gap_subset_qgnn.npz      ← quantum GNN input
│       ├── qm9_gap_subset_index.csv
│       ├── qm9_gap_subset_meta.json
│       └── qm9_gap_full_graphs.pt       all 130,745 graphs (built by notebook 04 on first run)
└── (papers: Therrien 2025 PDF, QIGNN and dipole-moment pages)
```

## 8. Results so far

All numbers are test MAE (eV) on the same 2,000 test molecules. Full details are in `QGNN_05_Comparison.ipynb`.

| Model | Params | Trained on | Random split | Scaffold split |
|---|---|---|---|---|
| **GINE** | 156,417 | full 104k (30 epochs) | **0.131** (0.130 on the full 13k test) | – |
| **GINE** | 156,417 | 10k subset | **0.173 ± 0.001** (3 seeds) | 0.316 ± 0.004 |
| GAT | 156,929 | 10k subset | 0.184 (1 seed) | – |
| GCN | 86,785 | 10k subset | 0.197 (1 seed) | – |
| Quantum GNN (4 layers) | 75 | 10k subset | 0.400 ± 0.020 | 0.468 ± 0.027 |
| Classical twin (TinyGNN) | 79 | 10k subset | 0.420 ± 0.022 | 0.471 ± 0.014 |
| Linear on mean atom features | 24 | 10k subset | 0.497 | – |
| Mean predictor | 1 | – | 1.067 | – |

What this says for the classical side:

- **The pipeline is validated:** GINE on the full data reaches 0.130 eV after only 30 CPU epochs, close to the literature 2D-GCN figure (≈ 0.12 eV), and was still improving when it stopped.
- **GINE > GAT > GCN** on the subset. Bond features matter: GCN, which ignores them, is the weakest.
- **New chemistry is harder:** on the scaffold split the GINE error is 1.8× higher. The small models only rise about 1.1–1.2×, but they start from a much higher error.
- **More data helps a lot:** 104k vs 10k training molecules gives 0.131 vs 0.173 eV.

The scaffold test set contains only 1 fluorine molecule, so per-element errors on that split are not meaningful.

## 9. Roadmap

1. ✅ EDA and preprocessing
2. ✅ Subset and graph dataset (random + scaffold splits)
3. ✅ Classical GNN baselines: GINE / GCN / GAT on the subset, GINE on the full data, scaffold split
4. ✅ Quantum GNN: built, tested against PennyLane, trained, ablations, 3 seeds × 2 splits
5. ✅ Comparison notebook (updates automatically from `results/`)
6. ⏳ QI-GNN: add 3D coordinates from PyG's QM9 set, plus distance/direction edge embeddings
7. ⏳ Longer training runs (GPU), plus 3 seeds for GCN/GAT and the full-data GINE
