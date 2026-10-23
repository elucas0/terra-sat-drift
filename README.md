# TerraSat Drift

TerraSat Drift is a research project for analyzing domain gap and data drift between real and simulated satellite imagery using [terratorch](https://github.com/IBM/terratorch) backbones (TerraMind). It covers the full pipeline from dataset simulation through model training to drift analysis.

## Repository Layout

```text
.
├── main.py / run.py                          # Entrypoints
├── terra_sat_drift/
│   ├── data_simulation/                      # Terratorch dataset simulation
│   │   ├── simulation_pipeline.py            #   End-to-end simulation orchestration
│   │   ├── simulation_config.py              #   Simulation parameters
│   │   ├── simulation_triplets.py            #   Triplet generation (real / simulated / label)
│   │   ├── degradation_ladder.py             #   Progressive degradation levels
│   │   ├── batch_simulate_sen1floods.py      #   Sen1Floods11 batch simulation
│   │   ├── batch_simulate_hls_burn_scars.py  #   HLS Burn Scars batch simulation
│   │   ├── phisat2_constants.py              #   φ-Sat-2 sensor constants
│   │   ├── phisat2_utils.py                  #   φ-Sat-2 simulation helpers
│   │   └── utils/                            #   Loaders, resampling, sanity checks
│   │
│   ├── dataset/                              # Datamodules & dataset definitions
│   │   ├── datamodule_*_triplets_lulc.py     #   Lightning DataModules (paired, S2B, real)
│   │   ├── dataset_*_triplets_lulc.py        #   Dataset classes (paired, S2B, base)
│   │   ├── constants.py                      #   Band names, class maps, normalisation stats
│   │   └── plot_utils.py                     #   Visualisation helpers
│   │
│   ├── drift_analysis/                       # Domain gap & data drift analysis
│   │   ├── drift_analysis.py                 #   Core drift metrics
│   │   ├── drift_pipeline.py                 #   Drift computation pipeline
│   │   ├── spectral_utils.py                 #   Band-level spectral statistics
│   │   ├── classification_drift_analyzer.py  #   Class stability & confidence drift
│   │   ├── segmentation_drift_analyzer.py    #   Segmentation-level drift
│   │   ├── sen1floods_drift_loader.py        #   Sen1Floods-specific drift loader
│   │   ├── reporting.py                      #   Console & JSON report formatting
│   │   ├── visualize_triplets_lulc.py        #   Triplet overlay visualisations
│   │   ├── plot_degradation_ladder_grid.py   #   Degradation ladder plots
│   │   ├── *.ipynb                           #   Analysis notebooks (embeddings, domain gap)
│   │   ├── figures/                          #   Generated figures (histograms, PDFs)
│   │   └── triplet_viz/                      #   Triplet overlay PNGs
│   │
│   └── model_tasks/                          # Train / test / validation scripts & modules
│       ├── baseline_module.py                #   Baseline Lightning module
│       ├── kd_module.py                      #   Knowledge-distillation module
│       ├── kd_contrastive_module.py          #   KD + contrastive learning module
│       ├── student_mobilenet.py              #   UNet student model definition
│       ├── phisatnet_student.py              #   PhisatNet student model definition
│       ├── losses/                           #   Loss functions (segmentation, contrastive)
│       ├── fine_tuning_tasks/                #   Training entry-points & notebooks
│       │   ├── kd_lulc.py                    #     KD training (TerraMind → UNet)
│       │   ├── kd_lulc_contrastive.py        #     Cross-sensor contrastive KD
│       │   ├── finetune_floods_student.py    #     Student flood-seg transfer probe
│       │   ├── finetune_floods_terramind.py  #     TerraMind flood-seg fine-tune
│       │   ├── train_baseline_phisat2.py     #     No-KD control baseline
│       │   ├── train_phisatnet_lulc.py       #     PhisatNet frozen-encoder LULC probe
│       │   ├── pretrain_lulc.py              #     TerraMind LULC pretraining
│       │   ├── pretrain_ssl.py               #     MAE-style SSL pretraining
│       │   ├── batch_probe_ladder.py         #     Degradation-ladder sweep orchestrator
│       │   ├── train_validate_floods*.py     #     Flood training scripts (real & simulated)
│       │   └── *.ipynb                       #     Training / inference notebooks
│       ├── decoder_tasks/                    #   Decoder-head training scripts
│       └── domain_adaptation/                #   MMD, DSBN & domain-adaptation training
│
├── scripts/                                  # Standalone utility scripts
├── notebooks/                                # Exploration & comparison notebooks
├── experiments/                              # Generated drift result JSONs
├── executables/                              # φ-Sat-2 simulator binaries (per-platform)
├── docs/                                     # MkDocs project documentation
└── mkdocs.yml                                # Docs site config
```

## Requirements

- Python ≥ 3.11
- Dependencies, see [pyproject.toml](pyproject.toml)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or with `uv`:

```bash
uv sync
```

## Documentation

Project docs are under `docs/` and built with MkDocs.

```bash
mkdocs serve   # local dev server
mkdocs build   # static site
```
