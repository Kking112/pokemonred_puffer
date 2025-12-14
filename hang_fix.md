# Hang Fix for Swarm Migration (v3 - FINAL)

## Problem
Training hangs indefinitely after swarm migration is triggered (after "New events..." is printed).

## Root Cause
The hang was in `self.vecenv.async_reset()` (line 482). The pufferlib `async_reset()` has a flush loop that blocks indefinitely waiting for workers to complete their steps. Since the vecenv was in an invalid state ("Call reset before stepping"), workers couldn't complete and the flush loop hung forever.

## Solution
**Removed `async_reset()` and wait loop entirely.**

The SQLite wrapper already handles state migration automatically:
1. Database UPDATE sets `reset=1` and stores new `pyboy_state`
2. Environments continue running normally
3. When episodes naturally end, `SqliteStateResetWrapper.reset()` reads the database
4. If `reset=1`, it loads the new state and sets `reset=0`

No forced reset needed - environments pick up new state on their next natural episode end.

## Changes Made

### `pokemonred_puffer/cleanrl_puffer.py`
- Removed `self.vecenv.async_reset()` call
- Removed the entire wait loop (polling for `reset=0`)
- Added explanatory comment and simple log message

## Expected Output
```
Satisified early stopping constraint for EVENT_BEAT_BROCK within 30 minutes. Event found in 27.0 minutes
    New events (9): (...)
[Swarm Migration] Database updated with new states for X environments
State migration to <path> complete
```

Training should continue without hanging.
