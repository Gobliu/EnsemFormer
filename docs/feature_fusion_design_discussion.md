# Feature Fusion Design Discussion for PhiGenerator

**Date:** 2026-02-26
**Context:** Reviewing the `MolecularFeatureFusion` class (formerly in `src/networks/phi_generator.py`) as a candidate approach for combining 1D pair distances, atom type embeddings, and bond type embeddings in the PhiGenerator module.

---

## Background

The PhiGenerator module needs to merge three input signals into a per-atom feature vector (phi):

| Input | Shape | Description |
|---|---|---|
| Pairwise distance | `[B, N, N]` | Euclidean distance between all atom pairs |
| Atom type embedding | `[N, E]` | Learnable vectors for H and O |
| Bond type embedding | `[N, N, E]` | Learnable vectors for directed bond types (H->O, O->H, H->H, no-bond) |

## The Proposed Approach (MolecularFeatureFusion, line 105)

The class uses:
1. **Gaussian RBF expansion** of the 1D distance into a high-dimensional vector
2. **Separate nn.Embedding layers** for atom types and bond types
3. **Concatenation + linear layer** to fuse all three features

## What Works Well

- **RBF expansion for distances** is the correct standard technique (used in SchNet, DimeNet, PaiNN). A raw scalar distance is too information-poor; Gaussian basis functions provide a "receptive field" for different distance scales.
- **Separate embeddings** for atoms and bonds keep categorical signals distinct before fusion.
- **SiLU activation** is a solid modern choice for the distance projection MLP.

## Issues Identified

### 1. Concatenation fusion is suboptimal

Concatenating `[dist_feat, atom_feat, bond_feat]` followed by a single linear layer forces the network to learn distance-chemistry interactions from scratch. In molecular GNNs, **multiplicative modulation** is preferred. Furthermore, adding `atom_feat + bond_feat` is commutative and loses directionality. The correct design concatenates *both* source and target atom embeddings with the directed bond embedding, fuses them through an MLP, then gates by distance:

```python
atom_bond = MLP(cat([atom_vec_i, atom_vec_j, bond_vec]))   # i->j != j->i
out = dist_feat * atom_bond
```

This encodes two inductive biases: (1) "the influence of a neighbor depends on how far away it is," and (2) the *pair* of atom types together with bond direction jointly determine the message. Including both `atom_vec_i` and `atom_vec_j` (not just the source) captures the full interaction context and generalizes naturally to new atom types.

### 2. Mean-pooling atom pairs loses directionality

The class averages embeddings of atom_i and atom_j (`a_embs.mean(dim=1)`), making the representation symmetric (i->j == j->i). This **conflicts with the existing PhiGenerator design**, which deliberately encodes directed bond types (H->O != O->H). Directionality information is lost.

**Better alternatives:** concatenate source/target embeddings, or use separate linear projections for each.

### 3. Does not integrate with PhiGenerator's tensor structure

PhiGenerator already computes `atom_vec [N, E]`, `bond_vec [N, N, E]`, and `dist [B, N, N]` in full pairwise tensor form. The MolecularFeatureFusion class expects flat `[N, 1]` distance inputs — it would require reshaping everything to use it, defeating the purpose.

### 4. Hardcoded RBF centers may have wrong units

`torch.linspace(0.0, 5.0, num_rbf)` assumes Angstrom units with a 5 A cutoff. Our water system uses GROMACS, which defaults to **nanometers**. The cutoff should be a constructor parameter with units matching the dataset.

## Recommended Architecture

Instead of the standalone MolecularFeatureFusion class, implement the fusion directly inside PhiGenerator using this pipeline:

```
Step 1: RBF-expand distances
    dist [B, N, N] -> RBF -> [B, N, N, K] -> Linear -> [B, N, N, E]

Step 2: Directional atom-bond fusion
    # atom_vec_i: [N, E] -> [N, 1, E],  atom_vec_j: [N, E] -> [1, N, E]
    # both broadcast to [N, N, E] before cat
    atom_bond = MLP(cat([atom_vec_i, atom_vec_j, bond_vec], dim=-1))   # [N, N, E]

Step 3: Multiplicative modulation by distance
    edge_feat = dist_feat * atom_bond          # [B, N, N, E]  (atom_bond broadcasts over B)

Step 4: Aggregate over neighbors
    phi = edge_feat.sum(dim=2)                 # [B, N, E]
```

**Key advantages:**
- Stays in the pairwise `[B, N, N]` tensor structure
- Preserves directed bond information: i->j and j->i produce different `atom_bond` values
- Captures full pairwise context: both source and target atom types enter the MLP
- Uses multiplicative gating (strong inductive bias for distance-dependent interactions)
- Generalizes to new atom types by extending the embedding table, not the architecture

## Resolution Status

All four issues have been resolved in the current implementation:

- [x] **Issue 1 (concatenation fusion):** Replaced with multiplicative modulation: `MLP(cat([atom_vec_i, atom_vec_j, bond_vec])) * dist_feat`
- [x] **Issue 2 (mean-pooling directionality):** Fixed — atom_i and atom_j are kept separate and concatenated, preserving directed bond information
- [x] **Issue 3 (tensor structure mismatch):** Fixed — fusion is implemented directly inside PhiGenerator, operating on `[B, N, N]` pairwise tensors
- [x] **Issue 4 (wrong units):** Fixed — RBF uses nanometers with configurable cutoff (default 1.1 nm); standalone `MolecularFeatureFusion` class removed
