# AGENTS.md — Curvant-ML Codebase Guide

## Quick Start

**Install & prepare data:**
```bash
pip install -e "[.dev]"
python scripts/preprocess_data.py --input data/<dataset>.parquet
```

**Run ML pipeline (common flags):**
```bash
python scripts/run.py --risco                    # binary safety classification
python scripts/run.py --isl                      # 3-class ISL + related regression
python scripts/run.py --velocidade               # critical speed regression
python scripts/run.py --multitarefa              # PyTorch multi-task MLP
python scripts/run.py --risco --otimizar --plot  # Optuna tuning + plots
```

**Interactive visualization:**
```bash
streamlit run app/main.py
```

## Core Concepts (short)
- 5-stage pipeline: preprocess → curve detection → characterization → feature extraction → ML training. See [src/pipeline.py](src/pipeline.py).
- Curve detection uses B-spline Frenet curvature; DNIT classes and `raio_min` clipping avoid ISL → ∞ ([src/curve_detection.py](src/curve_detection.py)).
- Risk taxonomy: three independent binary criteria (`manobra_accel`, `manobra_lateral`, `manobra_ziguezague`) and `manobra_combinado` (OR) — implemented in [src/characterization.py](src/characterization.py).
- Features grouped F1–F5; targets documented in [TARGETS_E_FEATURES.md](TARGETS_E_FEATURES.md) and [src/features.py](src/features.py).

## Key files (for agent use)
- [README.md](README.md) — project overview and examples
- [CLAUDE.md](CLAUDE.md) — detailed architecture, data notes, and pitfalls (primary reference for agents)
- [config.yaml](config.yaml) — all thresholds and hyperparameters
- [scripts/run.py](scripts/run.py) — CLI entry for experiments
- [scripts/preprocess_data.py](scripts/preprocess_data.py) — data cleaning
- [src/pipeline.py](src/pipeline.py) — orchestrator used by CLI and app
- [src/curve_detection.py](src/curve_detection.py)
- [src/characterization.py](src/characterization.py)
- [src/features.py](src/features.py)
- [src/models/tabular.py](src/models/tabular.py)
- [src/models/multitarefa.py](src/models/multitarefa.py)
- [app/main.py](app/main.py) — Streamlit UI

## Notes / Agent guidance
- There are no automated tests in the repository. Validate changes locally before proposing large refactors.
- Data is cached in `data/` after the first pipeline run; use `--rebuild` in `scripts/run.py` to force reprocessing when changing preprocessing or curve-detection parameters.
- Use the `_base_route` (strip `_p<N>`) grouping to avoid leakage when creating train/validation splits.
- Prefer linking to in-repo docs (CLAUDE.md, TARGETS_E_FEATURES.md, README.md) rather than copying large blocks into agent instructions.

## If you want more
- I can split agent guidance into focused instruction files (e.g., `AGENTS-ml.md`, `.github/copilot-instructions.md`) or create specialized skills for running experiments and committing results. Tell me which area to expand.
