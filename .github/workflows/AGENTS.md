<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# workflows

## Purpose
GitHub Actions CI/CD workflow definitions for automated testing and validation.

## Key Files

| File | Description |
|------|-------------|
| `workflow.yml` | Main CI workflow: runs `pytest` on push/PR to validate environment tests pass |

## For AI Agents

### Working In This Directory
- Standard GitHub Actions YAML syntax
- The workflow badge is displayed in `README.md`
- Tests run without the ROM file since PyBoy is mocked

### Common Patterns
- Uses `pip install -e '.[test]'` to install test dependencies
- Runs `pytest tests/` as the main test step

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
