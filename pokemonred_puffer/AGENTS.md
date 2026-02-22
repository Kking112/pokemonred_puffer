<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# pokemonred_puffer

## Purpose
Core source package containing the Pokemon Red RL environment, training loop, reward functions, policy networks, and gym wrappers. This is the main Python package — all source code lives here.

## Key Files

| File | Description |
|------|-------------|
| `train.py` | CLI entrypoint (`python -m pokemonred_puffer.train`): commands for `train`, `evaluate`, `autotune`, and `debug`. Dynamically loads rewards/wrappers/policies from config |
| `environment.py` | `RedGymEnv` — the core Gymnasium environment wrapping PyBoy. Handles game state reading, action execution, observation construction, auto-moves (cut, surf, strength), and episode management (~87KB) |
| `cleanrl_puffer.py` | `CleanPuffeRL` — PPO training loop with PufferLib integration. Handles rollout collection, GAE computation, policy updates, checkpointing, W&B logging, and swarm state synchronization (~44KB) |
| `eval.py` | Visualization utilities: generates heatmap overlays of agent exploration on the Kanto map using Numba JIT |
| `global_map.py` | Coordinate mapping utilities: converts local game coordinates (x, y, map_id) to global Kanto map positions using `map_data.json` |
| `resnet.py` | ResNet building blocks (BasicBlock, Bottleneck, ResNet) adapted from torchvision for potential policy architecture use |
| `c_gae.pyx` | Cython implementation of Generalized Advantage Estimation for faster training |
| `profile.py` | Performance profiling utilities for measuring training throughput |
| `sweep.py` | Hyperparameter sweep orchestration using CARBS (launch-sweep, launch-agent) |
| `events.json` | Game event flag definitions and metadata |
| `map_data.json` | Map coordinate data for all Kanto locations (used by `global_map.py`) |
| `pokered.sym` | Pokemon Red symbol table for memory address lookups |
| `kanto_map_dsv.png` | High-resolution Kanto map image used as overlay background |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `data/` | Game data enums and constants: events, maps, items, species, moves, tilesets (see `data/AGENTS.md`) |
| `rewards/` | Reward function implementations as `RedGymEnv` subclasses (see `rewards/AGENTS.md`) |
| `wrappers/` | Gymnasium wrappers for exploration decay, streaming, stats, and state management (see `wrappers/AGENTS.md`) |
| `policies/` | Neural network policy architectures (see `policies/AGENTS.md`) |
| `visualizations/` | Replay and visualization scripts (see `visualizations/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- `environment.py` is the largest and most complex file (~87KB). Read specific sections rather than the whole file
- `RedGymEnv` is the base class that reward environments inherit from. Modify rewards by subclassing in `rewards/`, not by editing `environment.py` directly
- New features are typically added via config-driven modules: create a new class in `rewards/`, `wrappers/`, or `policies/`, then reference it in `config.yaml`
- The `train.py` CLI uses Typer. Add new commands with `@app.command()` decorator

### Testing Requirements
- Run `pytest tests/` from the project root
- Environment tests mock PyBoy — the ROM file is not needed for testing
- For manual testing: `python -m pokemonred_puffer.train --config config.yaml --debug`

### Common Patterns
- Reward environments subclass `RedGymEnv` and override `get_game_state_reward()` returning a `dict[str, float]`
- Game memory is read via `self.read_m("symbol_name")` or `self.pyboy.memory[address]`
- Config is passed as `DictConfig` objects from OmegaConf
- Wrappers follow the standard `gymnasium.Wrapper` pattern and are composed in order from `config.yaml`

## Dependencies

### Internal
- `data/` — game constants used extensively by `environment.py` and `rewards/`
- `global_map.py` — used by `eval.py`, `cleanrl_puffer.py`, and exploration wrappers

### External
- `pyboy` — Game Boy emulation
- `pufferlib` — vectorized environment management and CleanRL framework
- `gymnasium` — RL environment API
- `torch` — neural network training
- `omegaconf` — configuration parsing
- `numba` — JIT for eval overlays

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
