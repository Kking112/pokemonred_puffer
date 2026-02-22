<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# tests

## Purpose
Unit tests for the Pokemon Red RL environment. Tests mock the PyBoy emulator to validate environment logic without requiring the ROM file.

## Key Files

| File | Description |
|------|-------------|
| `test_environment.py` | Tests for `RedGymEnv`: validates opponent level tracking, party management, and environment state updates |

## For AI Agents

### Working In This Directory
- Tests use `pytest` with `unittest.mock` for mocking PyBoy
- The `environment_fixture` creates a `RedGymEnv` with a mocked PyBoy instance
- Config is loaded from `config.yaml` in the project root via OmegaConf

### Testing Requirements
- Run: `pytest tests/`
- No ROM file needed — PyBoy is fully mocked
- Add new test files following the `test_*.py` naming convention

### Common Patterns
- Use `@pytest.fixture()` for environment setup
- Mock `read_m` to simulate game memory reads
- Use `@pytest.mark.parametrize` for testing multiple input combinations

## Dependencies

### Internal
- `pokemonred_puffer.environment` — the module under test

### External
- `pytest` — test framework
- `omegaconf` — config loading
- `unittest.mock` — mocking PyBoy

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
