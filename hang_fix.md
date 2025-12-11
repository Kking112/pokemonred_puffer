# Hang Fix for Swarm Migration (v2)

## Problem
Training script hangs indefinitely after swarm migration is triggered (after "New events..." is printed).

## Root Cause
The wait loop at lines 485-498 waits for all environments in `event_tracker.keys()` to have their `reset` flag cleared in the SQLite database. The loop hangs if:
1. `event_tracker` contains env_ids not in the database
2. Worker environments don't complete their reset within expected time
3. There's a lock contention issue with `DB_LOCK`

## Changes Made (v2)

### `pokemonred_puffer/cleanrl_puffer.py`

1. **Added timeout to wait loop** (60 seconds max)
   - Prevents infinite hang - training continues after timeout

2. **Added extensive debugging output**
   - Shows env_ids being waited on vs database env_ids
   - Detects mismatched env_ids
   - Logs progress of reset wait

3. **Handles missing env_ids**
   - If env_ids in event_tracker don't exist in database, they're skipped

## Debug Output
When swarm migration triggers, you'll see:
```
[Swarm Migration] Waiting for X envs to reset: [list]...
[Swarm Migration] Database has Y env_ids: [list]...
[Swarm Migration] Check 1: Z envs still pending reset
[Swarm Migration] Pending env_ids: [list]...
```

If it times out:
```
[Swarm Migration] TIMEOUT after 60.0s! N envs still pending.
[Swarm Migration] Forcing continue despite pending resets...
```

## Testing
Run training and let it reach the first EVENT_BEAT_BROCK. The debug output will help identify exactly why the wait loop isn't completing.
