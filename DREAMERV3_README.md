# DreamerV3 for Pokemon Red

This document describes the DreamerV3 implementation for Pokemon Red, including architecture, training, and usage.

## Overview

This repository implements DreamerV3, a state-of-the-art model-based reinforcement learning algorithm, for Pokemon Red. The implementation includes:

1. **Text Understanding**: A transformer-based text encoder that processes in-game text extracted directly from game memory
2. **World Model**: A Recurrent State-Space Model (RSSM) that learns to predict the environment dynamics
3. **Actor-Critic**: Policy and value networks trained via imagination
4. **Hybrid Mode**: Ability to use DreamerV3's world model as a representation encoder for PPO

## Architecture

### Text Extraction and Encoding

#### Text Extraction (`pokemonred_puffer/data/text.py`)

The `TextExtractor` class reads in-game text directly from Pokemon Red's memory:

- **Memory Addresses**:
  - `FONT_LOADED` (0xCFC4): Indicates if text is being displayed
  - `TEXT_BOX_ID` (0xD125): Current text box ID
  - `TEXT_BUFFER_1` (0xCD3D): Primary text buffer
  - `TEXT_BUFFER_2` (0xCE9D): Secondary text buffer

- **Character Decoding**: Pokemon Red uses a custom character encoding. The decoder maps byte values to characters (e.g., 0x80 = 'A', 0x81 = 'B')

- **Usage**:
```python
from pokemonred_puffer.data.text import TextExtractor

extractor = TextExtractor(pyboy_instance)
text_ids, decoded_text, is_displaying = extractor.get_current_text()
```

#### Text Encoder (`pokemonred_puffer/models/text_encoder.py`)

The `TransformerTextEncoder` processes text using:

- **Architecture**:
  - Character embeddings (256 vocab size for Pokemon Red's character set)
  - Sinusoidal positional encoding
  - Multi-head self-attention (configurable layers and heads)
  - Layer normalization and dropout

- **Output**: Fixed-dimensional text representation that is concatenated with visual observations

### World Model (`pokemonred_puffer/models/dreamer/world_model.py`)

The world model learns a latent representation of the environment using an RSSM.

#### Components

1. **Observation Encoder**: Processes multi-modal observations:
   - Screen pixels (CNN)
   - Visited mask (CNN)
   - In-game text (Transformer)
   - Game state (direction, map ID, levels, HP, events, etc.)

2. **RSSM (Recurrent State-Space Model)**:
   - **Deterministic state**: GRU-based recurrent state
   - **Stochastic state**: Discrete latent variables (categorical distribution)
   - **Prior**: Predicts next state from action and current state
   - **Posterior**: Infers state from observation

3. **Observation Decoder**: Reconstructs screen pixels from latent state

4. **Reward Predictor**: Predicts rewards from latent state

5. **Continue Predictor**: Predicts episode termination from latent state

#### RSSM Equations

```
# Recurrent model (deterministic transition)
h_t = f_θ(h_{t-1}, s_{t-1}, a_{t-1})

# Representation model (posterior)
s_t ~ q_θ(s_t | h_t, o_t)

# Transition predictor (prior)
ŝ_t ~ p_θ(ŝ_t | h_t)

# Observation model
ô_t = g_θ(h_t, s_t)

# Reward model
r̂_t = r_θ(h_t, s_t)

# Continue model
ĉ_t = c_θ(h_t, s_t)
```

### Actor-Critic (`pokemonred_puffer/models/dreamer/actor_critic.py`)

The policy and value functions are trained via imagination.

#### Actor

- **Architecture**: MLP with layer normalization and SiLU activations
- **Output**: Action distribution (categorical for discrete actions)
- **Training**: Maximizes imagined returns using policy gradient

#### Critic Ensemble

- **Architecture**: Multiple critic networks (default: 2-3)
- **Output**: Symlog-transformed value distribution
- **Training**: Minimizes temporal difference error
- **Target Network**: EMA-updated target for stability

#### Imagination Rollout

The actor-critic is trained on imagined trajectories:

1. Sample initial states from replay buffer
2. Use world model to imagine future trajectories
3. Compute λ-returns from imagined rewards
4. Update actor to maximize returns
5. Update critic to predict returns

### Replay Buffer (`pokemonred_puffer/models/dreamer/replay_buffer.py`)

Stores and samples sequences of experience.

- **Sequence-based**: Samples fixed-length sequences for recurrent training
- **Multi-environment**: Separate buffers for each parallel environment
- **Efficient**: Pre-allocated numpy arrays for fast access

## Training

### Pure DreamerV3 Training

```bash
# Use the DreamerV3 training script
uv run python pokemonred_puffer/dreamer_train.py \
  --config dreamer-config.yaml \
  --wandb \
  --device cuda
```

#### Training Loop

1. **Prefill**: Collect random experience to initialize replay buffer
2. **Main Loop**:
   - Collect experience from environment using current policy
   - Train world model on sequences from replay buffer
   - Train actor-critic via imagination
   - Log metrics and save checkpoints

#### Hyperparameters (dreamer-config.yaml)

```yaml
dreamer:
  algorithm: 'dreamerv3'
  
  # Replay buffer
  replay_capacity: 2_000_000
  sequence_length: 50
  batch_size: 16
  prefill_steps: 5000
  
  # Model architecture
  deter_dim: 1024          # Deterministic state dimension
  stoch_dim: 32            # Number of categorical variables
  stoch_classes: 32        # Classes per categorical variable
  
  # Training
  world_model_lr: 1e-4
  actor_lr: 3e-5
  critic_lr: 3e-5
  world_model_train_steps: 2
  actor_critic_train_steps: 2
  imagination_horizon: 15
  
  # Regularization
  kl_weight: 1.0
  kl_balance: 0.8
  free_nats: 1.0
  entropy_coef: 3e-4
```

### Hybrid Mode (DreamerV3 + PPO)

Use the DreamerV3 world model as a representation encoder with PPO for policy optimization.

```bash
# Train with hybrid mode
uv run python pokemonred_puffer/train.py \
  --config dreamer-ppo-config.yaml \
  --wandb
```

#### Configuration (dreamer-ppo-config.yaml)

```yaml
dreamer:
  algorithm: 'dreamer_ppo_hybrid'
  
  # World model architecture (same as pure DreamerV3)
  deter_dim: 512
  stoch_dim: 32
  stoch_classes: 32
  
  # Hybrid-specific
  freeze_world_model: false  # Set to true if using pre-trained world model
  policy_hidden_dim: 512
  policy_num_layers: 2
  
train:
  # Standard PPO settings
  total_timesteps: 10_000_000_000
  batch_size: 65536
  learning_rate: 0.0001
  use_rnn: true  # Uses RSSM as recurrent state
```

#### Advantages of Hybrid Mode

1. **Leverage existing infrastructure**: Uses PufferLib's proven PPO implementation
2. **Better sample efficiency**: World model provides learned representations
3. **Flexibility**: Can pre-train world model then fine-tune with PPO
4. **Debugging**: Easier to debug than full DreamerV3

## Environment Integration

The text extraction is integrated into `RedGymEnv`:

```python
# In pokemonred_puffer/environment.py

def _get_obs(self):
    # ... existing observation code ...
    
    # Add text observation
    text_ids, text_decoded, text_displaying = self.text_extractor.get_current_text()
    
    obs = {
        'screen': screen,
        'visited_mask': visited_mask,
        'text': text_ids,  # Added text observation
        # ... other observations ...
    }
    
    return obs
```

## Testing

Comprehensive tests are provided:

```bash
# Run all DreamerV3 tests
PYTHONPATH=. uv run pytest tests/test_dreamer/ -v

# Run specific component tests
PYTHONPATH=. uv run pytest tests/test_dreamer/test_text_extraction.py -v
PYTHONPATH=. uv run pytest tests/test_dreamer/test_world_model.py -v
PYTHONPATH=. uv run pytest tests/test_dreamer/test_actor_critic.py -v
PYTHONPATH=. uv run pytest tests/test_dreamer/test_replay_buffer.py -v
```

### Test Coverage

- **Text Extraction**: Character decoding, buffer reading, caching
- **World Model**: RSSM forward/backward, encoding, decoding
- **Actor-Critic**: Action sampling, value prediction, training loop
- **Replay Buffer**: Sequence sampling, multi-env support, wraparound

## Monitoring and Debugging

### Weights & Biases Integration

```python
wandb.init(
    project="pokemon-dreamerv3",
    config=config,
)
```

Key metrics logged:

- **World Model**:
  - `world_model/total_loss`: Combined world model loss
  - `world_model/reconstruction_loss`: Image reconstruction
  - `world_model/reward_loss`: Reward prediction
  - `world_model/continue_loss`: Episode termination prediction
  - `world_model/kl_loss`: KL divergence between prior and posterior

- **Actor-Critic**:
  - `actor_critic/actor_loss`: Policy loss
  - `actor_critic/critic_loss`: Value function loss
  - `actor_critic/entropy`: Policy entropy
  - `actor_critic/advantages_mean`: Mean advantage value
  - `actor_critic/returns_mean`: Mean imagined return

- **Collection**:
  - `episode/reward`: Episode total reward
  - `episode/length`: Episode length
  - `collection/total_reward`: Reward collected per iteration

### Checkpointing

Checkpoints are saved periodically and include:

```python
{
    'iteration': iteration,
    'global_step': global_step,
    'world_model': world_model.state_dict(),
    'actor': actor.state_dict(),
    'critic': critic.state_dict(),
    'text_encoder': text_encoder.state_dict(),
    'optimizers': {...},
}
```

Load checkpoint:

```bash
uv run python pokemonred_puffer/dreamer_train.py \
  --checkpoint checkpoints/dreamerv3/checkpoint_10000.pt \
  --config dreamer-config.yaml
```

## Performance Considerations

### Memory Usage

- **Replay Buffer**: ~2-8 GB RAM (depends on capacity)
- **Model**: ~500 MB - 2 GB GPU VRAM (depends on architecture size)
- **Batch Processing**: ~4-8 GB GPU VRAM during training

### Recommended Hardware

Tested on:
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **RAM**: 64 GB
- **CPU**: AMD Ryzen 9

Minimum requirements:
- **GPU**: NVIDIA GPU with ≥8 GB VRAM (RTX 3070 or better)
- **RAM**: 16 GB
- **CPU**: Modern multi-core CPU

### Optimization Tips

1. **Use AMP (Automatic Mixed Precision)**:
   ```yaml
   dreamer:
     use_amp: true
   ```

2. **Adjust batch sizes based on GPU memory**:
   ```yaml
   dreamer:
     batch_size: 8  # Reduce if OOM
     num_envs: 4    # Reduce if OOM
   ```

3. **Use gradient accumulation for large effective batch sizes**:
   ```yaml
   train:
     accumulation_steps: 4
   ```

4. **Enable compilation (PyTorch 2.0+)**:
   ```yaml
   train:
     compile: true
     compile_mode: reduce-overhead
   ```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solution**: Reduce batch size, num_envs, or model dimensions

```yaml
dreamer:
  batch_size: 8  # Down from 16
  num_envs: 2    # Down from 4
  deter_dim: 512 # Down from 1024
```

#### 2. Training instability

**Solution**: Adjust learning rates, KL weight, or entropy coefficient

```yaml
dreamer:
  world_model_lr: 5e-5  # Lower learning rate
  kl_weight: 0.5        # Reduce KL penalty
  entropy_coef: 1e-3    # Increase exploration
```

#### 3. World model not learning

**Symptoms**: Reconstruction loss not decreasing

**Solution**:
- Check observation preprocessing (normalization)
- Increase world model capacity
- Train world model longer before actor-critic

#### 4. Policy not improving

**Symptoms**: Episode reward not increasing

**Solution**:
- Ensure world model is learning well first
- Increase imagination horizon
- Adjust discount factor and λ
- Check reward scale (use reward normalization if needed)

## Future Improvements

Potential enhancements:

1. **Pre-trained Text Encoder**: Use a pre-trained language model for text understanding
2. **Hierarchical RL**: Multi-level policies for long-horizon tasks
3. **Curriculum Learning**: Gradually increase task difficulty
4. **Multi-task Learning**: Train on multiple Pokemon games simultaneously
5. **Model Distillation**: Distill world model into smaller model for faster inference
6. **Prioritized Experience Replay**: Sample important sequences more frequently
7. **Intrinsic Motivation**: Add curiosity-driven exploration bonuses

## References

1. [DreamerV3 Paper](https://arxiv.org/abs/2301.04104)
2. [Original PufferRL Pokemon Red Experiment](https://github.com/thatguy11325/pokemonred_puffer)
3. [World Models](https://worldmodels.github.io/)
4. [RSSM (Recurrent State-Space Model)](https://arxiv.org/abs/1811.04551)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pokemon_dreamerv3,
  title={DreamerV3 for Pokemon Red},
  author={Your Name},
  year={2025},
  howpublished={\\url{https://github.com/yourusername/pokemonred_puffer}},
}
```

## License

Same as the original pokemonred_puffer repository.

