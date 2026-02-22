<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# visualizations

## Purpose
Replay and visualization tools for analyzing agent behavior from recorded training data.

## Key Files

| File | Description |
|------|-------------|
| `actions_replay.py` | Replays recorded action sequences to visualize agent decision-making |
| `coordinates_replay.py` | Replays recorded coordinate data to visualize agent movement patterns on the map |
| `player.png` | Player sprite asset used in visualization overlays |

## For AI Agents

### Working In This Directory
- These are standalone analysis scripts, not part of the training pipeline
- They consume data produced by the `CoordinatesWriter` wrapper and training logs
- Visualization outputs are for offline analysis and debugging

### Common Patterns
- Scripts read from the `runs/` output directory
- Use OpenCV and NumPy for image processing

## Dependencies

### Internal
- `pokemonred_puffer/global_map.py` — coordinate mapping for map visualization

### External
- `opencv-python` — image rendering
- `numpy` — array operations

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
