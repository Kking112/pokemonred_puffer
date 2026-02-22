<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# poke-baseline

## Purpose
Reinforcement learning framework for training AI agents to play Pokemon Red using the PyBoy Game Boy emulator, PufferLib for vectorized environment management, and CleanRL-based PPO training. The project supports configurable reward shaping, exploration wrappers, hyperparameter sweeps via CARBS, and experiment tracking with Weights & Biases.

## Key Files

| File | Description |
|------|-------------|
| `config.yaml` | Primary configuration: environment settings, training hyperparameters, reward weights, wrapper definitions, and policy architecture |
| `config_base.yaml` | Base configuration template |
| `config_run_penalty.yaml` | Configuration for the run penalty/reward experiment (current branch) |
| `pyproject.toml` | Python project metadata, dependencies, and Ruff linting/formatting rules |
| `requirements.txt` | Pip-installable dependency list |
| `red.gb` | Pokemon Red ROM file (required, not in repo) |
| `sweep-config.yaml` | CARBS hyperparameter sweep configuration |
| `wild-battles-sweep-config.yaml` | Sweep config variant for wild battle experiments |
| `cut-config.yaml` | Sweep config variant for cut-related experiments |
| `test.yaml` | Test configuration for CI |
| `hang_fix.md` | Documentation on diagnosing and fixing training hangs |
| `.pre-commit-config.yaml` | Pre-commit hooks (Ruff formatting/linting) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `pokemonred_puffer/` | Main source package: environment, training, rewards, policies, wrappers (see `pokemonred_puffer/AGENTS.md`) |
| `pyboy_states/` | Pre-saved PyBoy game states for various progression checkpoints (see `pyboy_states/AGENTS.md`) |
| `tests/` | Unit tests for the environment (see `tests/AGENTS.md`) |
| `assets/` | Project assets like logos (see `assets/AGENTS.md`) |
| `.github/` | GitHub Actions CI/CD workflows (see `.github/AGENTS.md`) |
| `runs/` | Training run output directories (checkpoints, logs) |
| `wandb/` | Weights & Biases experiment tracking data |
| `video/` | Video recordings captured during training |

## For AI Agents

### Working In This Directory
- Install with `pip install -e .` (or `pip install -e '.[dev]'` for development)
- Configuration is YAML-based via OmegaConf. The `config.yaml` file is the central configuration surface
- Rewards, wrappers, and policies are referenced in `config.yaml` by `module_name.ClassName` keys
- The ROM file `red.gb` must be present but is not committed to version control
- Python 3.10+ required. Uses Ruff for formatting (line length 100, double quotes, spaces)

### Testing Requirements
- Run tests: `pytest tests/`
- Tests mock PyBoy to avoid requiring the ROM file
- Pre-commit hooks enforce Ruff formatting and linting

### Common Patterns
- Config-driven architecture: rewards, wrappers, and policies are dynamically loaded via `importlib` based on `config.yaml` keys
- Training entrypoint: `python -m pokemonred_puffer.train train`
- Debug mode: `python -m pokemonred_puffer.train --config config.yaml --debug`
- Environments use PufferLib's vectorized environment pool for parallel training

## Dependencies

### External
- `pufferlib` (1.0 fork) - Vectorized RL environment management and CleanRL integration
- `pyboy` (>=2) - Game Boy emulator for running Pokemon Red
- `torch` (>=2.4) - Neural network training
- `wandb` - Experiment tracking and logging
- `omegaconf` - YAML configuration management
- `gymnasium` - RL environment interface
- `numba` - JIT compilation for performance-critical map overlay code
- `typer` / `tyro` - CLI argument parsing

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
