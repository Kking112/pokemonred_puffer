<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# data

## Purpose
Game data definitions for Pokemon Red: enums, constants, memory addresses, and lookup tables extracted from the ROM's data structures. These modules map raw game memory into typed Python enums and constants used throughout the environment and reward system.

## Key Files

| File | Description |
|------|-------------|
| `events.py` | `EventFlags` bitfield struct (ctypes) mapping all ~320 game event flags from RAM address 0xD747. Defines `REQUIRED_EVENTS`, `EVENTS`, and provides `get_event()`/`get_events()` for reading event completion state (~104KB) |
| `map.py` | `MapIds` enum for all Kanto map IDs, plus `MAP_ID_COMPLETION_EVENTS` linking maps to their completion event flags |
| `species.py` | `Species` enum mapping Pokemon species IDs to names |
| `moves.py` | `Moves` enum mapping move IDs to names |
| `items.py` | `Items` enum with item IDs. Defines `HM_ITEMS`, `KEY_ITEMS`, `REQUIRED_ITEMS`, `USEFUL_ITEMS`, and `MAX_ITEM_CAPACITY` |
| `bag.py` | Bag/inventory data structures and utilities |
| `party.py` | `PartyMons` — party Pokemon data structures for reading party state from memory |
| `flags.py` | `Flags` — miscellaneous game flags |
| `tm_hm.py` | `TmHmMoves` enum for TM/HM move IDs. Defines `CUT_SPECIES_IDS`, `STRENGTH_SPECIES_IDS`, `SURF_SPECIES_IDS` — which Pokemon can learn each HM |
| `field_moves.py` | `FieldMoves` — field move mechanics and lookups |
| `missable_objects.py` | `MissableFlags` — flags for items/objects that can be picked up once |
| `strength_puzzles.py` | `STRENGTH_SOLUTIONS` — step-by-step boulder push solutions for each strength puzzle |
| `elevators.py` | `NEXT_ELEVATORS` — elevator floor transition mappings |
| `tilesets.py` | `Tilesets` enum categorizing map tilesets (overworld, gym, facility, etc.) for exploration reward differentiation |

## For AI Agents

### Working In This Directory
- These are mostly data-only modules with enums and constants — they rarely need modification
- `events.py` is the largest file (~104KB) due to the exhaustive event flag bitfield definition
- When adding new game data, follow the existing enum pattern and use the Pokemon Red RAM map as reference
- Memory addresses come from the `pokered` disassembly: https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map

### Testing Requirements
- Data modules are implicitly tested through environment tests
- Enum values must match the actual ROM memory layout exactly

### Common Patterns
- Enums inherit from Python's `enum.IntEnum` for direct comparison with memory values
- Bitfield structs use `ctypes.LittleEndianStructure` for mapping RAM flag bytes
- Constants like `REQUIRED_EVENTS` and `REQUIRED_ITEMS` define game completion milestones used by reward functions

## Dependencies

### Internal
- Used by `pokemonred_puffer/environment.py` — imported extensively for game state interpretation
- Used by `pokemonred_puffer/rewards/` — for event/item reward calculations

### External
- `pyboy` — for `symbol_lookup` memory address resolution
- `ctypes` — for bitfield struct definitions

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
