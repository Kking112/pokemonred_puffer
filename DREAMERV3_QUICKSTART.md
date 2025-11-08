# DreamerV3 Quick Start Guide

This guide will help you get started with DreamerV3 for Pokemon Red in 5 minutes.

## Prerequisites

1. Python 3.11+
2. CUDA-capable GPU (recommended: ≥8GB VRAM)
3. `uv` package manager installed

## Installation

```bash
# Clone the repository
cd pokemonred_puffer

# Install dependencies
uv sync

# Install the package in development mode
uv pip install -e .
```

## Quick Training Examples

### Example 1: Pure DreamerV3 (Recommended for Research)

Train using the full DreamerV3 algorithm with text understanding:

```bash
uv run python pokemonred_puffer/dreamer_train.py \
  --config dreamer-config.yaml \
  --wandb \
  --device cuda
```

This will:
- Extract and encode in-game text using a Transformer
- Learn a world model of Pokemon Red
- Train an actor-critic via imagination
- Log metrics to Weights & Biases

**Expected behavior**:
- Initial episodes will be random exploration
- After ~5k steps (prefill), training begins
- World model reconstruction loss should decrease
- Episode rewards should gradually increase

### Example 2: Hybrid Mode (DreamerV3 + PPO)

Use DreamerV3's representations with PPO's policy optimization:

```bash
uv run python pokemonred_puffer/train.py \
  train \
  --config dreamer-ppo-config.yaml \
  --wandb
```

**Advantages**:
- Leverages existing PufferLib infrastructure
- Often more stable than pure DreamerV3
- Can use pre-trained world model

### Example 3: Debug Mode (Small Scale)

For testing and debugging:

```bash
uv run python pokemonred_puffer/dreamer_train.py \
  --config dreamer-config.yaml \
  --device cpu
```

Use debug configuration in `config.yaml`:

```yaml
debug:
  dreamer:
    algorithm: 'dreamerv3'
    num_envs: 1
    batch_size: 4
    sequence_length: 10
    prefill_steps: 100
    num_iterations: 100
```

## Configuration

### Minimal Configuration

Create `my_experiment.yaml`:

```yaml
# Inherit from base config
defaults:
  - config

# Override DreamerV3 settings
dreamer:
  algorithm: 'dreamerv3'
  
  # Scale based on your GPU
  num_envs: 4              # Parallel environments
  batch_size: 16           # Sequences per batch
  
  # Training
  world_model_train_steps: 2
  actor_critic_train_steps: 2
  
  # Logging
  log_interval: 10
  checkpoint_interval: 500
  checkpoint_dir: 'my_checkpoints'

# W&B logging
wandb:
  project: my-pokemon-experiment
  entity: your-wandb-username
```

Run with:

```bash
uv run python pokemonred_puffer/dreamer_train.py \
  --config my_experiment.yaml \
  --wandb
```

## Monitoring Training

### Weights & Biases Dashboard

Key metrics to watch:

1. **World Model Health**:
   - `world_model/reconstruction_loss` → Should decrease to ~0.1-0.3
   - `world_model/kl_loss` → Should stabilize around 1-10
   - `world_model/reward_loss` → Should decrease

2. **Policy Performance**:
   - `episode/reward` → Should increase over time
   - `episode/length` → Episode duration
   - `actor_critic/entropy` → Should slowly decrease (exploration → exploitation)

3. **Training Speed**:
   - `training/iteration_time` → Time per training iteration
   - `training/global_step` → Total environment steps

### Terminal Output

```
[Iteration 100]
  Global step: 20000
  Replay buffer: 20000/2000000
  WM Loss: 0.234
  Actor Loss: -12.5
  Critic Loss: 3.4
  Collection reward: 145.2
```

## Checkpoints and Resuming

### Save Checkpoint

Checkpoints are automatically saved every `checkpoint_interval` iterations to:

```
checkpoints/dreamerv3/checkpoint_<iteration>.pt
```

### Load Checkpoint

Resume training from a checkpoint:

```bash
uv run python pokemonred_puffer/dreamer_train.py \
  --config dreamer-config.yaml \
  --checkpoint checkpoints/dreamerv3/checkpoint_10000.pt \
  --wandb
```

### Export for Evaluation

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/dreamerv3/checkpoint_10000.pt')

# Extract components
world_model_state = checkpoint['world_model']
actor_state = checkpoint['actor']
critic_state = checkpoint['critic']
text_encoder_state = checkpoint['text_encoder']
```

## Testing Your Setup

### 1. Test Text Extraction

```bash
PYTHONPATH=. uv run pytest tests/test_dreamer/test_text_extraction.py -v
```

Expected: Most tests should pass (8/11 core tests pass)

### 2. Test World Model

```bash
PYTHONPATH=. uv run pytest tests/test_dreamer/test_world_model.py::TestRSSM -v
```

Expected: All RSSM tests should pass (6/6)

### 3. Test Actor-Critic

```bash
PYTHONPATH=. uv run pytest tests/test_dreamer/test_actor_critic.py -v
```

Expected: All tests should pass (13/13)

### 4. Test Replay Buffer

```bash
PYTHONPATH=. uv run pytest tests/test_dreamer/test_replay_buffer.py -v
```

Expected: All tests should pass (10/10)

### 5. Quick Training Test

Run a short training run to verify everything works:

```bash
# Create a test config
cat > test_config.yaml << EOF
defaults:
  - config

dreamer:
  algorithm: 'dreamerv3'
  num_envs: 1
  batch_size: 4
  sequence_length: 10
  prefill_steps: 50
  num_iterations: 10
  log_interval: 1
EOF

# Run test training
uv run python pokemonred_puffer/dreamer_train.py \
  --config test_config.yaml \
  --device cuda
```

Expected output:
- Should complete prefill phase
- Should run 10 training iterations
- No errors or crashes

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size and/or number of environments

```yaml
dreamer:
  batch_size: 8   # Down from 16
  num_envs: 2     # Down from 4
```

### Issue 2: Import Error

```
ModuleNotFoundError: No module named 'pokemonred_puffer'
```

**Solution**: Install package and set PYTHONPATH

```bash
uv pip install -e .
export PYTHONPATH=/path/to/pokemonred_puffer:$PYTHONPATH
```

### Issue 3: Slow Training

**Solution**: Enable optimizations

```yaml
dreamer:
  use_amp: true  # Automatic mixed precision

train:
  compile: true  # PyTorch 2.0 compilation
```

### Issue 4: World Model Not Learning

**Symptoms**: Reconstruction loss stays high

**Solution**: Check observation preprocessing

```python
# Verify observations are properly normalized
# Screen should be in [0, 1] or [-1, 1] range
```

## Next Steps

1. **Read Full Documentation**: See `DREAMERV3_README.md` for detailed architecture and training information

2. **Experiment with Hyperparameters**: Try different values for:
   - `imagination_horizon`: Length of imagined rollouts
   - `deter_dim` and `stoch_dim`: World model capacity
   - Learning rates: `world_model_lr`, `actor_lr`, `critic_lr`

3. **Monitor and Iterate**: Use W&B to track experiments and identify improvements

4. **Pre-train World Model**: Train world model first, then fine-tune with PPO

5. **Contribute**: Share your results and improvements!

## Getting Help

- **Documentation**: `DREAMERV3_README.md`
- **Tests**: `tests/test_dreamer/`
- **Issues**: Check for existing issues or create a new one
- **Discussions**: Start a discussion in the repository

## Quick Reference

### File Structure

```
pokemonred_puffer/
├── pokemonred_puffer/
│   ├── data/
│   │   └── text.py                    # Text extraction
│   ├── models/
│   │   ├── text_encoder.py            # Transformer text encoder
│   │   └── dreamer/
│   │       ├── world_model.py         # World model (RSSM)
│   │       ├── actor_critic.py        # Policy and value networks
│   │       ├── replay_buffer.py       # Experience replay
│   │       └── utils.py               # Utility functions
│   ├── dreamer_train.py               # Main training script
│   └── models/dreamer_ppo_policy.py   # Hybrid PPO policy
├── config.yaml                        # Base configuration
├── dreamer-config.yaml                # DreamerV3 configuration
├── dreamer-ppo-config.yaml            # Hybrid mode configuration
├── tests/
│   └── test_dreamer/                  # Tests for all components
├── DREAMERV3_README.md                # Full documentation
└── DREAMERV3_QUICKSTART.md            # This file
```

### Key Commands

```bash
# Train DreamerV3
uv run python pokemonred_puffer/dreamer_train.py --config dreamer-config.yaml

# Train Hybrid (DreamerV3 + PPO)
uv run python pokemonred_puffer/train.py train --config dreamer-ppo-config.yaml

# Run tests
PYTHONPATH=. uv run pytest tests/test_dreamer/ -v

# Evaluate checkpoint
uv run python pokemonred_puffer/eval.py --checkpoint checkpoints/dreamerv3/checkpoint_10000.pt
```

Happy training! 🎮🚀

