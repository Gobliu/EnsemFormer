# EnsemFormer Memory

## Project Overview
- **EnsemFormer**: molecular property prediction using ensembles of 3D conformers
- **First model**: CycloFormer — predicts cyclic peptide membrane permeability (PAMPA)
- Architecture: N conformers → shared GNN encoder → conformer tokens → Transformer → MLP head → scalar

## File Structure (as of 2026-03-02)
```
src/
  utils.py            — to_device, rank_zero_only, init_distributed, get_local_rank,
                        using_tensor_cores, get_split_sizes, get_next_version,
                        print_parameters_count, xavier_normal_small_init_,
                        xavier_uniform_small_init_
  loggers.py          — CSVLogger, TensorBoardLogger, LoggerCollection
  callbacks.py        — EarlyStoppingCallback, AllMetricsCallback, BaseCallback
  data_module.py      — DataModule (abstract), _get_dataloader (DDP-aware)
  featurization.py    — load_ensemble_from_smiles, load_ensemble_from_pdb,
                        featurize_mol, get_atom_features, _featurize_pdb_file,
                        one_hot_vector, Molecule, MolDataset
  dataset.py          — ConformerEnsembleMolecule, ConformerEnsembleDataset,
                        conformer_collate_fn, ConformerEnsembleDataModule
  trainer.py          — Trainer, save_state, load_state
  networks/
    egnn_encoder.py   — EGNNEncoder, E_GCL, get_edges_batch, mean_pool_atoms
    cpmp_encoder.py   — CPMPEncoder (all CPMP internal classes included)
    cycloformer.py    — CycloFormerModule, ConformerTransformerEncoder, MLPHead
models/
  Wrapper.py          — Module (abstract base, keep as-is)
  models.py           — LEGACY (broken OldCode imports, do not use)
scripts/
  main_train.py       — full training entrypoint
config/
  default.yaml        — canonical config schema
Main.py               — delegates to scripts/main_train.main()
```

## Key Design Decisions
- GNN encoder shared across all conformers; efficient batching via reshape (B,N_conf,N_atoms,F) → (B*N_conf,N_atoms,F)
- CPMP: dense atom matrices; EGNN: flat node list + get_edges_batch
- conformer_collate_fn returns: node_feat (B,N_conf,N_atoms,F), adj, dist, atom_mask, conformer_mask, target
- conformer_mask: True = real conformer; key_padding_mask = ~conformer_mask (inverted for Transformer)
- CycloFormerModule.model is nn.ModuleList([gnn_encoder, conformer_encoder, head])
- Config is a dict (not argparse); use types.SimpleNamespace(**config['training']) inside train_one_epoch

## Atom Feature Dimension
- d_atom = 25 (one_hot_formal_charge=False) or 27 (True)
- With dummy node: shape is (N_atoms+1, F+1) — dummy flag appended

## OldCode Reference (do not migrate further unless needed)
- OldCode/src/training.py: original Trainer (REGISTRY removed in new trainer.py)
- OldCode/cpmp/data/: PAMPA CSV datasets (training data)
- OldCode/se3-transformer/: SE3T backbone (deferred)
- OldCode/egnn/: EGNN source (migrated to src/networks/egnn_encoder.py)
- OldCode/cpmp/model/transformer.py: CPMP source (migrated to src/networks/cpmp_encoder.py)

## docs/ Status (cleaned 2026-03-02)
Kept: AGENTS.md, MULTI_GPU_USAGE.md, training_loop.md, feature_fusion_design_discussion.md
Deleted: data_flow.md, fig_*.py, fig_*.png (all LUFNet-specific)
