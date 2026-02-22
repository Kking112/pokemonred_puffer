<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# rewards

## Purpose
Reward function implementations for the Pokemon Red RL environment. Each reward class subclasses `RedGymEnv` and overrides `get_game_state_reward()` to define how the agent is rewarded. Reward classes form an inheritance chain of increasingly sophisticated reward shaping strategies.

## Key Files

| File | Description |
|------|-------------|
| `baseline.py` | Main reward module (~25KB) containing the reward class hierarchy: `BaselineRewardEnv` -> `TeachCutReplicationEnv` -> `TeachCutReplicationEnvFork` -> `CutWithObjectRewardsEnv` -> `CutWithObjectRewardRequiredEventsEnv` -> `ObjectRewardRequiredEventsMapIds` -> `ObjectRewardRequiredEventsMapIdsFieldMoves`. Each adds new reward signals (events, exploration, cut usage, required items, tileset-aware exploration, field moves) |
| `run_wrapper.py` | `RunPenaltyRewardEnv` — extends `ObjectRewardRequiredEventsMapIds` with penalties for running from battles and rewards for winning trainer battles (current branch feature) |
| `__init__.py` | Package initialization |

## For AI Agents

### Working In This Directory
- **To add a new reward**: Create a new class that subclasses an existing reward env (typically `ObjectRewardRequiredEventsMapIds` or `ObjectRewardRequiredEventsMapIdsFieldMoves`), override `get_game_state_reward()`, then add a config entry in `config.yaml` under the `rewards:` section keyed as `module_name.ClassName`
- Reward weights are configured in `config.yaml`, not hardcoded — `self.reward_config` holds the weights dict
- `get_game_state_reward()` returns a `dict[str, float]` where keys are reward component names and values are the weighted reward amounts
- Call `super().get_game_state_reward()` to inherit parent rewards, then add/override specific components

### Testing Requirements
- Test reward logic by running in debug mode: `python -m pokemonred_puffer.train debug -r module.ClassName`
- Verify reward components appear in W&B logs during training

### Common Patterns
- Read game memory with `self.read_m("symbol_name")` for named symbols or `self.pyboy.memory[address]` for raw addresses
- Track state between steps using instance variables (reset in `reset()`)
- Use `self.reward_config.get("key", default)` for configurable reward weights
- The inheritance chain builds progressively: each child adds reward signals on top of its parent

## Dependencies

### Internal
- `pokemonred_puffer/environment.py` — `RedGymEnv` base class
- `pokemonred_puffer/data/events.py` — event flag constants (`EVENTS`, `REQUIRED_EVENTS`)
- `pokemonred_puffer/data/items.py` — item constants (`REQUIRED_ITEMS`, `USEFUL_ITEMS`)
- `pokemonred_puffer/data/tilesets.py` — tileset categories for exploration differentiation

### External
- `numpy` — reward computation
- `omegaconf` — config access

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
