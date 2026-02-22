# Plan: Consolidate Training Outputs into Single Directory

## Requirements Summary

Currently, a training run creates **two separate directories**:
1. `<project_root>/<YYYYMMDD-HHMMSS>/` — event state archives (`.state` files + `desc.txt`)
2. `<project_root>/runs/pokemon-red-<uuid>/` — model checkpoints (`model_*.pt`, `trainer_state.pt`)

**Goal**: Consolidate so that a single training run creates **one directory**:
```
runs/pokemon-red-<uuid>/
├── model_000200.pt
├── model_000225.pt
├── trainer_state.pt
└── logging/
    ├── 4817614215452469064/
    │   ├── desc.txt
    │   └── -5652840396851042231.state
    ├── -820819082767692181/
    │   ├── desc.txt
    │   └── <hash>.state
    └── ...
```

No datetime-stamped directory should be created in the project root.

## Acceptance Criteria

1. Running `uv run -m pokemonred_puffer.train train` creates NO new directory in the project root
2. All `.state` files and `desc.txt` files are saved under `runs/<exp_id>/logging/<hash>/`
3. Model checkpoints (`model_*.pt`, `trainer_state.pt`) continue to be saved at `runs/<exp_id>/` (unchanged)
4. The `logging/` subdirectory is created automatically when `archive_states` is True
5. The print statement still announces where states will be archived (updated path)
6. No changes to config.yaml schema are required (the existing `archive_states` and `data_dir` config keys remain)

## Implementation Steps

All changes are in a single file: `pokemonred_puffer/cleanrl_puffer.py`

### Step 1: Build the checkpoint base path early in `__post_init__`

**File**: `pokemonred_puffer/cleanrl_puffer.py`
**Lines**: 209-212

The `archive_path` is currently set to a datetime string in the project root. It needs to point into the checkpoint directory instead. However, the checkpoint path (`config.data_dir / config.exp_id`) is not constructed until `save_checkpoint()`. We need to:

1. Construct the base run directory path (`runs/<exp_id>`) early in `__post_init__` and store it as `self.run_path`
2. Create `self.run_path` directory immediately (it will be needed for both checkpoints and state archiving)
3. Set `self.archive_path` to `self.run_path / "logging"` instead of the datetime string

**Replace lines 209-212**:
```python
# BEFORE:
if self.config.archive_states:
    self.archive_path = pathlib.Path(datetime.now().strftime("%Y%m%d-%H%M%S"))
    self.archive_path.mkdir(exist_ok=False)
    print(f"Will archive states to {self.archive_path}")

# AFTER:
self.run_path = pathlib.Path(os.path.join(self.config.data_dir, self.config.exp_id))
self.run_path.mkdir(parents=True, exist_ok=True)
if self.config.archive_states:
    self.archive_path = self.run_path / "logging"
    self.archive_path.mkdir(exist_ok=True)
    print(f"Will archive states to {self.archive_path}")
```

**Key details**:
- `parents=True` ensures the `runs/` parent directory is created if it doesn't exist
- `exist_ok=True` on `run_path` because `save_checkpoint` may also try to create it (and that's fine)
- `exist_ok=True` on `archive_path` for the same reason (idempotent)

### Step 2: Update `save_checkpoint()` to reuse `self.run_path`

**File**: `pokemonred_puffer/cleanrl_puffer.py`
**Lines**: 725-729

Since we now create the run directory in `__post_init__`, `save_checkpoint()` should reuse it instead of reconstructing the path.

**Replace lines 727-729**:
```python
# BEFORE:
path = os.path.join(config.data_dir, config.exp_id)
if not os.path.exists(path):
    os.makedirs(path)

# AFTER:
path = str(self.run_path)
```

The directory is guaranteed to exist because `__post_init__` already created it. Converting `self.run_path` (a `pathlib.Path`) to `str` keeps compatibility with the `os.path.join` calls that follow on lines 732, 746.

### Step 3: Verify no other code references the old datetime directory pattern

**Verification**: The `datetime.now().strftime("%Y%m%d-%H%M%S")` pattern for directory creation only appears at line 210 of `cleanrl_puffer.py`. The `coords_writer.py` file also uses a datetime pattern but that is for a separate logging purpose (coordinate tracking) and is unrelated to state archiving.

The `self.archive_path` usage at lines 299-306 (in `evaluate()`) does NOT need any changes — it already writes to `self.archive_path / str(hash(key))`, and since we changed what `self.archive_path` points to, the state files will automatically go to the right place.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `exp_id` not set when `__post_init__` runs | `exp_id` is set in `train.py:setup()` at line 181 BEFORE `CleanPuffeRL` is instantiated at line 454 — confirmed safe |
| `os` module not imported for `os.path.join` in new code | `os` is already imported at the top of `cleanrl_puffer.py` — confirmed |
| `save_checkpoint` called before `__post_init__` | Impossible — `__post_init__` runs at object construction, `save_checkpoint` is called during the training loop |
| Old datetime directories still being created by other code | Only one place creates them (line 210) — confirmed via grep |

## Verification Steps

1. **Static check**: Grep the codebase for `strftime` to confirm no other datetime directory creation exists
2. **Static check**: Verify `self.run_path` is used consistently in both `__post_init__` and `save_checkpoint`
3. **Runtime test**: Run a short training session and verify:
   - No datetime directory appears in the project root
   - `runs/<exp_id>/logging/` contains state subdirectories with `desc.txt` and `.state` files
   - `runs/<exp_id>/` contains `model_*.pt` and `trainer_state.pt`
