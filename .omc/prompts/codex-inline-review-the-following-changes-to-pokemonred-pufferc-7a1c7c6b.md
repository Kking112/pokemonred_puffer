Review the following changes to `pokemonred_puffer/cleanrl_puffer.py` in a Pokemon Red reinforcement learning training project. The goal is to consolidate training outputs so that state archives (.state files and desc.txt) are saved under `runs/<exp_id>/logging/` instead of a separate datetime-stamped directory in the project root.

## Changes Made (3 edits in one file):

### 1. Removed unused import (line 3):
```diff
-from datetime import datetime
```

### 2. `__post_init__` method — replaced datetime archive path with run_path/logging:
```diff
+        self.run_path = pathlib.Path(os.path.join(self.config.data_dir, self.config.exp_id))
+        self.run_path.mkdir(parents=True, exist_ok=True)
         if self.config.archive_states:
-            self.archive_path = pathlib.Path(datetime.now().strftime("%Y%m%d-%H%M%S"))
-            self.archive_path.mkdir(exist_ok=False)
+            self.archive_path = self.run_path / "logging"
+            self.archive_path.mkdir(exist_ok=True)
             print(f"Will archive states to {self.archive_path}")
```

### 3. `save_checkpoint()` method — reuse self.run_path instead of reconstructing:
```diff
     def save_checkpoint(self):
         config = self.config
-        path = os.path.join(config.data_dir, config.exp_id)
-        if not os.path.exists(path):
-            os.makedirs(path)
+        path = str(self.run_path)
```

## Key context:
- `self.config.data_dir` is "runs" (from config.yaml)
- `self.config.exp_id` is set in train.py:setup() as `f"pokemon-red-{str(uuid.uuid4())[:8]}"` BEFORE CleanPuffeRL is instantiated
- `self.archive_path` is used in `evaluate()` at lines 299-306 to write state files: `self.archive_path / str(hash(key))` — this code was NOT changed, it automatically uses the new path
- There is also a reference to `self.archive_path` at line 499 in a print statement for swarm migration that uses it as a display path — this also works correctly with the new path
- `os` and `pathlib` are already imported at the top of the file

## Verify:
1. Are there any race conditions or ordering issues?
2. Is the `run_path` guaranteed to be available when needed?
3. Are there any missed references to the old pattern?
4. Could `archive_states=False` cause issues (run_path is still created but archive_path is not set)?