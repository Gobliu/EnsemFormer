# EnsemFormer

**EnsemFormer** is a framework for molecular property prediction using ensembles of 3D conformers. The core idea is to treat a set of 3D structures (e.g. from an MD trajectory or conformational search) as a sequence of tokens, processed by a Transformer encoder to produce a single molecular-level representation.

## Current Model: CycloFormer

The first instantiation of EnsemFormer is **CycloFormer**, targeting **cyclic peptide membrane permeability** prediction (regression).

### Architecture

```
Conformer 1 ──┐
Conformer 2 ──┤  [3D GNN encoder]  →  conformer embeddings (tokens)
   ...        │                              ↓
Conformer N ──┘                    [Transformer encoder]
                                    (conformers as tokens)
                                              ↓
                                   [Readout: CLS or mean pool]
                                              ↓
                                         [MLP head]
                                              ↓
                                    Permeability value (regression)
```

- **3D GNN encoder**: Swappable backbone (EGNN, SE(3)-Transformer, etc.)
- **Conformer aggregation**: Transformer encoder over conformer embeddings
- **Task**: Regression (continuous permeability value, e.g. PAMPA)
- **Input**: Ensemble of 3D conformers per cyclic peptide

## Vision

EnsemFormer is designed to be domain-agnostic. While CycloFormer targets cyclic peptides, the framework can be extended to other molecular property prediction tasks where conformational diversity is important.
