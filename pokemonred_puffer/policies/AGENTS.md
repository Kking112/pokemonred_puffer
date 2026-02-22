<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-21 | Updated: 2026-02-21 -->

# policies

## Purpose
Neural network policy architectures for the RL agent. Policies process game observations (screen pixels, event flags, map data) and output action probabilities and value estimates for PPO training.

## Key Files

| File | Description |
|------|-------------|
| `multi_convolutional.py` | `MultiConvolutionalPolicy` (~11KB) — the primary policy network. Uses lazy convolutional layers (32->64->64 channels) to process the screen observation, with separate networks for optional global map input. Outputs are fed through a linear layer to produce actor (action logits) and value estimates. `MultiConvolutionalRNN` wraps it with an LSTM for temporal reasoning |
| `__init__.py` | Package initialization |

## For AI Agents

### Working In This Directory
- **To add a new policy**: Create a new `nn.Module` subclass, implement `forward()` returning `(hidden, value)`, then add a config entry in `config.yaml` under `policies:` keyed as `module_name.ClassName`
- Policies use `nn.Lazy*` layers to automatically infer input dimensions from the observation space
- The policy handles two-bit encoding of screen pixels (4 grayscale values mapped to 0/85/153/255)
- If `use_rnn: True` in config (default), the policy is wrapped in `MultiConvolutionalRNN` (LSTM) then `pufferlib.frameworks.cleanrl.RecurrentPolicy`
- The `forward()` method processes a composite observation dict with screen, event flags, and optionally global map data

### Testing Requirements
- Policy changes are validated by running a short training session
- Check that `torch.compile` works with the policy (the training loop uses `compile: True` by default)

### Common Patterns
- Screen observation: processed through convolutional layers with ReLU activations
- Additional inputs (badges, items, events): concatenated with screen features before the final linear layer
- `one_hot()` is a custom implementation because `torch.nn.functional.one_hot` cannot be traced by torch.compile
- Hidden size (default 512) is configurable via `config.yaml`

## Dependencies

### Internal
- `pokemonred_puffer/data/events.py` — `EVENTS_IDXS` for event flag indexing
- `pokemonred_puffer/data/items.py` — `Items` enum for item observation processing
- `pokemonred_puffer/environment.py` — `PIXEL_VALUES` for screen decoding

### External
- `torch` — neural network modules
- `pufferlib` — `LSTMWrapper`, `GymnasiumPufferEnv` interface, and `nativize_dtype`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
