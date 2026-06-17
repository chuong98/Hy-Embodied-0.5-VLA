Build / test / lint

- Install (recommended):
  - uv-based virtualenv (preferred):
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - uv sync  # materializes .venv with pinned wheels (uses uv.lock)
  - pip fallback (smoke/dev):
    - python -m pip install -r requirements.txt

- Quick smoke / single-run checks:
  - python scripts/quick_start.py           # fast smoke test using a released checkpoint
  - python test_install.py                  # simple end-to-end model load + forward pass
  - Render one episode (single-table):
    - python scripts/vis_umi_episode.py -t table_000 -e 666

- Training & evaluation (examples / single-run):
  - Single-table fast iteration:
    - export TABLE_NAME=table_001
    - bash scripts/train_table_vlm.sh
  - Fine-tune / distributed launches (multi-node):
    - set CHIEF_IP and INDEX environment variables; then run
    - bash scripts/train_umi_vlm.sh or bash scripts/train_robotwin_umi.sh
  - Evaluation (single-run):
    - export ROBOTWIN_DIR=/path/to/RoboTwin
    - export CKPT_PATH=tencent/Hy-Embodied-0.5-VLA-RoboTwin
    - bash scripts/eval_robotwin_test.sh  # quick regression
    - bash scripts/eval_robotwin_full.sh  # full sweep (high-cost)

- Norm stats (single-run):
  - python scripts/compute_norm_lance.py --lance-source <HF-or-local-root> --output norm_stats.pkl

- Lint: no repository-provided lint task or config detected. Use your preferred linter/formatters if needed.

High-level architecture (big picture)

- Top-level package: hy_vla/
  - modeling_hy_vla.py, modeling_dual_tower.py — model classes: HyVLA, dual-tower VLM + action expert.
  - configuration_hy_vla.py — model config loader.
  - space_time_attention.py — compact memory (interleaved temporal-spatial attention) used by the ViT backbone.
  - data/ — Lance, HDF5, and dataset utilities (LanceTableReader, dataset wrappers).
  - config/ — hydra/yaml training configs (base.yaml, zero2.json and dataset-specific YAMLs).
  - hunyuan_vl_mot/ — vendored fallback of the upstream Hy-Embodied transformers fork (used if remote git dependency is unavailable).

- Scripts: training/eval/visualization wrappers live under scripts/ (train_*.sh, eval_*.sh, vis_umi_episode.py, compute_norm_*.py, quick_start.py).

- robotwin_eval/: adapter and wrapper code to integrate Hy-VLA policies into RoboTwin evaluation and deployment (policy_wrapper.py, deploy_policy.py, eval.sh). Eval scripts auto-symlink this folder into RoboTwin's policy path.

Key repository conventions and gotchas (repo-specific)

- Environment & toolchain:
  - Python 3.10–3.12 expected; CUDA 12.x and PyTorch >=2.7 recommended.
  - uv is the supported environment manager. uv.sync materializes wheels listed in pyproject.toml and routes torch/flash-attn to prebuilt wheels via tool.uv.sources.

- Dependencies & vendor fallback:
  - transformers is a git-direct reference to a pinned upstream commit. If that URL is unreachable, code falls back to hy_vla/hunyuan_vl_mot/ which registers the same Auto* classes.

- Configuration & runtime flags:
  - Training uses Hydra-configured YAMLs in hy_vla/config/.
  - Multi-node training expects CHIEF_IP and INDEX env vars used by the provided bash wrappers.
  - Evaluation scripts expect ROBOTWIN_DIR and CKPT_PATH for RoboTwin integration.

- Data format & precomputation:
  - Released dataset uses Lance tables (LeRobot schema). hy_vla.data contains LanceTableReader and fallback HDF5 readers.
  - Normalization statistics are required for training; generate with scripts/compute_norm_lance.py or compute_norm_hdf5.py if using custom data.

- Checkpoint contents & inference:
  - Released checkpoints include tokenizer.json, vlm_config_dict, chat_template.jinja and norm_stats.pkl; quick_start and test_install rely on snapshot_download to fetch these assets.
  - For inference, enable video encoder via policy.enable_video_encoder_if_needed() before moving to device.

- Evaluation integration:
  - scripts/eval_* and robotwin_eval/ are designed to be runnable as-is; eval scripts create a symlink so RoboTwin can discover the policy adapter without manual edits.

- Resource expectations:
  - Many training and evaluation workflows assume large GPU counts and long runtimes. Use the single-table fast-iteration scripts for local development.

Files and places to check when troubleshooting

- hy_vla/train.py  — entrypoint for training-related imports and assumptions
- pyproject.toml / requirements.txt — pinned runtime and training dependencies (torch, flash-attn routing info)
- hy_vla/config/  — hydra config overrides; useful for quick experiment edits
- scripts/compute_norm_* and scripts/vis_umi_episode.py — common pre-run steps that often surface path/format issues

Notes on what was incorporated

- Key sections from README.md and pyproject.toml were summarized (installation, quick-start, training/eval scripts, uv usage and torch/flash-attn routing).
- No repository-provided linter or AI-assistant rules files (CLAUDE.md, AGENTS.md, .cursorrules, etc.) were found to incorporate.

Summary

Created .github/copilot-instructions.md with: install/run commands (single-run examples), high-level architecture summary, and repo-specific conventions (uv usage, transformers vendor fallback, Lance data and norm-stat precompute).

Would you like adjustments or additional coverage (examples for running a single training iteration, recommended linter config, or documentation snippets for Hydra overrides)?
