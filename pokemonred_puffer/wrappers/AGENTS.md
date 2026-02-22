<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# wrappers

## Purpose
Gymnasium wrappers that compose on top of the reward environment to add cross-cutting functionality: exploration map decay, episode statistics, live streaming, coordinate logging, and state synchronization. Wrappers are stacked in order as defined in `config.yaml`.

## Key Files

| File | Description |
|------|-------------|
| `exploration.py` | Exploration reward management (~10KB). Contains `DecayWrapper` (exponential decay of exploration maps by tileset category), `OnResetExplorationWrapper` (resets exploration maps periodically), `MaxLengthWrapper` (LRU-based coordinate capacity limit), and `OnResetLowerToFixedValueWrapper` (resets exploration to fixed values) |
| `episode_stats.py` | `EpisodeStatsWrapper` — tracks cumulative episode return and length, emits info dicts at log frequency intervals or episode boundaries |
| `stream_wrapper.py` | `StreamWrapper` — streams agent coordinates to the Pokemon Red Map Visualizer via WebSocket (`wss://transdimensional.xyz/broadcast`) |
| `async_io.py` | `AsyncWrapper` — asynchronous I/O wrapper using multiprocessing queues for inter-environment communication during swarm resets |
| `coords_writer.py` | `CoordinatesWriter` — writes agent coordinates to disk files at configurable frequency for offline analysis |
| `sqlite.py` | `SqliteStateResetWrapper` — uses a shared SQLite database for swarm state synchronization: environments share save states when milestones are reached, enabling population-wide reset to the best-progressing agent |
| `__init__.py` | Package initialization |

## For AI Agents

### Working In This Directory
- **To add a new wrapper**: Create a class inheriting from `gymnasium.Wrapper`, implement `step()` and optionally `reset()`, then add it to `config.yaml` under the `wrappers:` section
- Wrappers are applied in order (top to bottom) from the config. Order matters — e.g., `StreamWrapper` should wrap before `DecayWrapper`
- Some wrappers mutate the underlying environment directly (e.g., `DecayWrapper` modifies `env.seen_coords`) — this is intentional to save memory but should be done carefully
- Wrapper config is passed as a `DictConfig` to the constructor

### Testing Requirements
- Wrapper behavior is tested through full environment integration
- For manual testing, add the wrapper to the `empty` wrapper list in `config.yaml` and run debug mode

### Common Patterns
- Access the unwrapped environment via `self.env.unwrapped`
- Read game state from `self.env.unwrapped.read_m()`
- Wrappers that need periodic behavior use step counters (e.g., `self.env.unwrapped.step_count`)
- SQLite and async wrappers enable swarm training where environments synchronize state

## Dependencies

### Internal
- `pokemonred_puffer/environment.py` — `RedGymEnv` is the base environment being wrapped
- `pokemonred_puffer/global_map.py` — coordinate mapping used by exploration wrappers

### External
- `gymnasium` — `gym.Wrapper` base class
- `websockets` — WebSocket client for streaming
- `sqlite3` — shared state database for swarm synchronization
- `omegaconf` — wrapper configuration
- `pufferlib` — utilities for info dict processing

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
