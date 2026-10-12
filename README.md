# Classical vs Quantum Graph Neural Networks for Molecular Property Prediction (QM9)

Predicting the **HOMO–LUMO gap** of small organic molecules with:

- **classical graph neural networks** (GINE, GCN, GAT), and
- a **quantum graph neural network**: a 9-qubit variational circuit with one qubit per atom and entangling gates along the chemical bonds, simulated exactly.

All models use the same molecules, splits, target and metric (MAE in eV). The comparison also reports parameter counts, training time, a scaffold split for out-of-distribution generalisation, and ablations that test what the quantum circuit actually uses.

---

## Results

Test MAE in eV on the **same 2,000 test molecules**. Mean ± std over 3 seeds where seeds = 3.

| Model | Trainable params | Training data | Random split | Scaffold split |
|---|---:|---|---:|---:|
| GINE | 156,417 | full train set (104,595) | **0.131** ¹ | – |
| GINE | 156,417 | 10k subset | **0.173 ± 0.001** | 0.316 ± 0.004 |
| GAT (v2) | 156,929 | 10k subset | 0.184 | – |
| GCN | 86,785 | 10k subset | 0.197 | – |
| **Quantum GNN** (9 qubits, 4 layers) | **75** | 10k subset | **0.400 ± 0.020** | 0.468 ± 0.027 |
| Classical twin (same size) | 79 | 10k subset | 0.420 ± 0.022 | 0.471 ± 0.014 |
| Linear model on mean atom features | 24 | 10k subset | 0.497 | – |
| Mean predictor | 1 | – | 1.067 | – |

¹ 0.130 eV on the full 13,075-molecule test set, after 30 epochs on a CPU. The literature value for a 2D GCN is ≈ 0.12 eV (Therrien et al., 2025).

**Quantum ablations** (random split, seed 0):

| Variant | Params | Test MAE (eV) |
|---|---:|---:|
| Full quantum GNN | 75 | 0.379 |
| One ZZ weight for all bond types | 63 | 0.409 |
| No entanglement (ZZ gates removed) | 59 | 0.535 |
| Plain 8-qubit circuit on 8 PCA features (no graph) | 19 | 0.839 |

### Key findings

1. **The classical pipeline is validated.** GINE on the full data reaches 0.130 eV in 30 CPU epochs, close to published 2D-GNN results, and was still improving when it stopped.
2. **Bond features matter for classical GNNs.** GINE and GAT, which use bond features, beat GCN, which only sees the connectivity.
3. **The quantum GNN learns real structure.** It beats the linear baseline. Removing entanglement raises its error by about 40%, and dropping the graph from the circuit roughly doubles it.
4. **At equal size, quantum and classical tie.** The 75-parameter quantum GNN (0.400 eV) and a 79-parameter classical message-passing model (0.420 eV) are within seed-to-seed variation of each other. The classical one trains in about 3 s instead of about 150 s.
5. **Scale wins overall.** The 156k-parameter GINE has 2.3× lower error than the quantum GNN on the same 10k molecules, and adding more data helps further.
6. **New chemistry is harder.** On the scaffold split, GINE's error rises 1.8×; the small models rise 1.1–1.2× from a much higher starting error.

---

## Method

### Data
- **QM9** (MoleculeNet CSV): 133,885 molecules with ≤ 9 heavy atoms (C, N, O, F), properties computed with DFT (B3LYP/6-31G(2df,p)).
- **Cleaning:** removed invalid and duplicate SMILES and the 3,054 molecules that failed QM9's geometry consistency check, leaving **130,745 molecules**.
- **Target:** HOMO–LUMO gap converted to eV and standardised using the training set (mean 6.858 eV, std 1.284 eV).
- **Leakage control:** HOMO and LUMO (gap = LUMO − HOMO) and every other DFT property are excluded from the inputs. Inputs come only from molecular structure.
- **Split:** 80 / 10 / 10 random split (seed 42).
- **10k subset:** 10,000 / 2,000 / 2,000 molecules, drawn *inside* each original split. Sampling is stratified on gap decile × contains-fluorine, so the subset matches the full data in gap distribution, element mix and size. Because the subset test set lies inside the full test set, every model can be scored on the same 2,000 molecules.
- **Scaffold split** over the same 14,000 molecules: molecules sharing a Bemis–Murcko scaffold stay in the same split, and no scaffold appears in both train and test.

### Molecular graphs
Heavy-atom graphs (hydrogens become a node feature), so every molecule has **≤ 9 nodes**.
- **Node features (23):** element, degree, number of H, hybridisation, aromatic, in ring, formal charge.
- **Edge features (6):** bond type (single, double, triple, aromatic), conjugated, in ring.

### Classical GNNs
Each model embeds the atoms (23 → 128), applies 4 message-passing layers (GINEConv, GCNConv or GATv2Conv) with residual connections and batch norm, mean-pools over atoms, and predicts with an MLP head. Training: Adam (lr 1e-3), MSE loss, learning rate halved on validation plateau, early stopping on validation MAE.

### Quantum GNN
One qubit per heavy atom (9 qubits; smaller molecules are padded). Each of the L layers applies:

```
encode    RY(θ_i) RZ(φ_i) on atom i      (θ, φ) = Linear(23 → 2)(atom features), shared by all atoms
entangle  IsingZZ(w[bond type, layer])   on every bonded pair; non-bonded pairs get angle 0 = identity
mix       RX(α_l) RY(β_l)                on every qubit, trainable, shared
```

**Readout:** mean ⟨Z_i⟩ over real atoms and mean ⟨Z_i Z_j⟩ over bonds → `Linear(2 → 1)`.

- **Permutation invariant by construction:** weights are shared across atoms and set by bond type, and all ZZ gates commute.
- **Data re-uploading:** atom angles are encoded again in every layer.
- **The mix layer is required:** without a non-commuting rotation after the ZZ gates, entanglement cannot change ⟨Z⟩.

**Simulation:** a batched statevector simulator written in PyTorch. The whole entangling layer reduces to one diagonal phase per basis state. It matches a gate-by-gate **PennyLane** reference to ~1e-15 in both values and gradients, and runs about 80× faster. Unit tests cover atom relabelling, padding, bond sensitivity and the need for the mix layer.

**Classical twin:** a 79-parameter message-passing network with the same inputs and readout, used to compare quantum and classical models at equal size.

---

## Repository structure

```
├── QM9_EDA_Preprocessing.ipynb          1. EDA, cleaning, features, split, scaling
├── QGNN_01_Subset_GraphDataset.ipynb    2. 10k subset, graphs, scaffold split
├── QGNN_02_QuantumGNN.ipynb             3. quantum circuit: built and verified against PennyLane
├── QGNN_03_QuantumGNN_Training.ipynb    4. quantum training: depth sweep, ablations, seeds, scaffold
├── QGNN_04_ClassicalGNN.ipynb           5. GINE / GCN / GAT on the subset and the full data
├── QGNN_05_Comparison.ipynb             6. combined tables and plots (reads results/)
├── qgnn_lib.py                          quantum GNN, fast simulator, classical twin, training loop
├── qm9_graphs.py                        SMILES → PyG graph featuriser
├── QGNN_Project_Guide.md                longer walkthrough of the project
├── qm9.csv                              raw data (MoleculeNet)
├── processed/
│   ├── qm9_clean.csv                    130,745 cleaned molecules
│   ├── qm9_gap_eV_processed.npz         descriptors, fingerprints, PCA angles, split indices
│   ├── qm9_gap_eV_transforms.joblib     fitted scalers and PCA
│   └── graphs/
│       ├── qm9_gap_subset_graphs.pt     14k PyG graphs + random and scaffold splits
│       ├── qm9_gap_subset_qgnn.npz      the same molecules as padded 9-qubit arrays
│       ├── qm9_gap_subset_index.csv     molecule ↔ row ↔ split lookup
│       └── qm9_gap_subset_meta.json     feature names, target scaling, seed
└── results/
    ├── quantum_results.csv
    ├── classical_results.csv
    ├── trivial_baselines.csv
    └── quantum_config.json
```

## How to run

```bash
pip install -r requirements.txt
```

Run the notebooks in order: `QM9_EDA_Preprocessing` → `QGNN_01` → `QGNN_02` → `QGNN_03` → `QGNN_04` → `QGNN_05`. Each one reads the files written by the previous one.

**Data:** the large data files are not in the repo (see `.gitignore`). `QM9_EDA_Preprocessing` downloads `qm9.csv` from MoleculeNet automatically and writes `processed/`. `QGNN_01` then builds the graph files. The small subset files (`qm9_gap_subset_qgnn.npz`, the index CSV and the metadata) are included, so notebooks 02 and 03 run without the preprocessing step.

Approximate runtimes on a 2-core CPU:

| Notebook | Runtime |
|---|---|
| 01 | ~30 s |
| 02 | ~2 min |
| 03 | ~40 min |
| 04 | ~30 min (the full-data GINE is ~12 min of this) |
| 05 | seconds |

Training budgets (`EPOCHS`, `SUBSET_EPOCHS`, `FULL_EPOCHS`) are set at the top of notebooks 03 and 04. Raise them on a GPU.

Tested with Python 3.13, PyTorch 2.14, PyTorch Geometric 2.8, RDKit 2026.09 and PennyLane 0.45. The graph file `qm9_gap_subset_graphs.pt` is saved with PyTorch; if it does not load on another version, re-run `QGNN_01` (~30 s).

## Limitations

- **Short training:** the quantum and twin models were capped at 25 epochs and most were still improving. Classical runs were capped at 60 (subset) and 30 (full data) epochs. All numbers would improve with longer training.
- **Simulation only:** quantum results use exact statevector simulation with no shot noise or hardware noise.
- **Number of seeds:** GCN, GAT, the full-data GINE and the ablations use a single seed. Seed-to-seed spread for the quantum GNN is about ±0.02 eV.
- **Fluorine:** the scaffold test set contains only 1 fluorine molecule, so per-element errors on that split are not meaningful.
- **2D graphs only:** no 3D geometry is used yet. 3D models reach much lower errors (e.g. PaiNN ≈ 0.05 eV).

## Roadmap

- [ ] **Quantum-inspired GNN:** 3D coordinates from PyG's QM9, with distance and direction edge embeddings
- [ ] Longer training on a GPU, and 3 seeds for every model
- [ ] Shot-noise and hardware-noise experiments for the quantum GNN, plus a barren-plateau (gradient variance vs depth) study

## References

- Therrien, Sargent & Voznyy. *Using GNN property predictors as molecule generators.* Nature Communications 16:4301 (2025). doi:10.1038/s41467-025-59439-1
- Chanana. *Molecular property prediction toward materials science applications using quantum-inspired graph neural networks.* Springer Proceedings in Physics 369 (2026). doi:10.1007/978-981-92-4005-0_18
- Li et al. *Dipole moment prediction with graph neural networks and physics-informed models.* Chemical Papers 80 (2026). doi:10.1007/s11696-026-04657-0
- Verdon et al. *Quantum Graph Neural Networks.* arXiv:1909.12264 (2019)
- Mernyei, Meichanetzidis & Ceylan. *Equivariant Quantum Graph Circuits.* ICML (2022)
- Ramakrishnan et al. *Quantum chemistry structures and properties of 134 kilo molecules.* Scientific Data 1:140022 (2014)
