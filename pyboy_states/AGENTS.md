<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# pyboy_states

## Purpose
Pre-saved PyBoy emulator states representing various game progression checkpoints. These are loaded at environment initialization and used for state-based training resets and swarm synchronization.

## Key Files

| File | Description |
|------|-------------|
| `init.state` | Very beginning of the game |
| `home.state` | Player at home |
| `fast_text_start.state` | Game start with fast text enabled |
| `Bulbasaur.state` | After choosing Bulbasaur as starter |
| `Charmander.state` | After choosing Charmander as starter |
| `Squirtle.state` | After choosing Squirtle as starter |
| `cut.state` | After obtaining HM01 Cut |
| `mtmoon.state` | At Mt. Moon |
| `rocktunnel.state` | At Rock Tunnel |
| `pokeflute.state` | After obtaining the Poke Flute |
| `seafoam.state` | At Seafoam Islands |
| `victory_road.state` | At Victory Road |
| `beat_champion.state` | After beating the Champion |
| `beat_lance_1.state` | After beating Lance |
| `game_corner.state` | At the Game Corner |

## For AI Agents

### Working In This Directory
- State files are binary PyBoy snapshots — do not edit manually
- The `init_state` config key in `config.yaml` selects which state to load (without `.state` extension)
- New states are generated during training when `archive_states: True` is set in config
- States are used by the swarm mechanism to synchronize environments when milestones are reached

### Common Patterns
- States are loaded via `PyBoy.load_state()` in `environment.py`
- The `state_dir` config key points to this directory

## Dependencies

### Internal
- Used by `pokemonred_puffer/environment.py` — loaded during `RedGymEnv.__init__` and `reset()`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
