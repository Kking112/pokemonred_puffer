# DreamerV3 Implementation Notes

This document describes key implementation decisions, challenges, and design choices made while implementing DreamerV3 for Pokemon Red.

## Text Understanding

### Design Decision: Memory Address Extraction vs. OCR

**Chosen Approach**: Direct memory address extraction

**Rationale**:
1. **Accuracy**: 100% accurate character extraction vs. potential OCR errors
2. **Efficiency**: Minimal computational overhead
3. **Reliability**: No dependency on visual rendering quality
4. **Simplicity**: Direct access to game state without ML inference

**Implementation Details**:

The Pokemon Red ROM uses a custom character encoding stored at specific memory addresses:
- `FONT_LOADED` (0xCFC4): Boolean indicating text display
- `TEXT_BOX_ID` (0xD125): Identifies which text box is active
- `TEXT_BUFFER_1/2` (0xCD3D/0xCE9D): Character arrays

Character mapping:
```python
# Examples from Pokemon Red's character table
0x80 -> 'A'
0x81 -> 'B'
0xF4 -> '0'
0x00 -> <END>
0x4E -> '\n'
```

**Challenges**:
- Multiple text buffers can be active
- Text persists in memory after display ends
- Special characters require custom handling

**Solution**:
- Cache last seen text
- Check `FONT_LOADED` and `TEXT_BOX_ID` for display state
- Try primary buffer first, fall back to secondary

### Text Encoder Architecture

**Chosen Approach**: Transformer Encoder

**Alternatives Considered**:
1. LSTM: Simpler but less powerful for long-range dependencies
2. CNN: Fast but poor at capturing sequential structure
3. Pre-trained LM: More powerful but adds complexity and size

**Rationale for Transformer**:
1. **Attention**: Captures relationships between distant words
2. **Parallelism**: Faster training than RNNs
3. **Modularity**: Easy to swap with pre-trained encoder later
4. **Scalability**: Can increase layers/heads as needed

**Architecture**:
```python
Input: [batch, seq_len] character IDs
  ↓
Embedding: [batch, seq_len, embed_dim]
  ↓
Positional Encoding: Sinusoidal
  ↓
Transformer Layers: Multi-head self-attention + FFN
  ↓
Pooling: Mean over sequence
  ↓
Output: [batch, output_dim] fixed representation
```

**Parameters**:
- Vocab size: 256 (Pokemon Red character set)
- Embed dim: 128
- Num heads: 4
- Num layers: 2
- Output dim: 128

**Design Notes**:
- Used mean pooling instead of [CLS] token (no pre-training)
- LayerNorm + Dropout for regularization
- Sinusoidal positional encoding (no learned positions)

## World Model (RSSM)

### Discrete vs. Continuous Latents

**Chosen Approach**: Discrete latents (categorical distributions)

**Rationale** (from DreamerV3 paper):
1. **Expressiveness**: Discrete latents can represent complex, multi-modal distributions
2. **Stability**: Avoids posterior collapse issues of continuous VAEs
3. **Efficiency**: Straight-through gradients avoid high-variance REINFORCE

**Implementation**:
```python
# Stochastic state: 32 categorical variables with 32 classes each
stoch_dim = 32
stoch_classes = 32
# Total: 32 * 32 = 1024-dimensional one-hot encoded state
```

**Comparison**:
- **Continuous (DreamerV2)**: Gaussian distribution, can suffer from posterior collapse
- **Discrete (DreamerV3)**: Categorical distribution, more stable but higher dimensional

### RSSM State Representation

The RSSM maintains two components:

1. **Deterministic State (h)**: GRU-based recurrent state
   - Captures temporal dependencies
   - Updated every step: `h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])`

2. **Stochastic State (s)**: Discrete latent variables
   - Captures uncertainty and multi-modality
   - **Prior**: `p(s_t | h_t)` - Prediction from deterministic state
   - **Posterior**: `q(s_t | h_t, o_t)` - Inference from observation

**Training**:
- Minimize KL divergence between prior and posterior
- KL balancing: `KL_loss = 0.8 * KL(q||p) + 0.2 * KL(p||q)`
- Free nats: Don't penalize KL below threshold (encourages expressiveness)

### Multi-Modal Observation Encoding

**Challenge**: Pokemon Red has diverse observation types:
- **Visual**: Screen pixels (72x80x1), visited mask (72x80x1)
- **Text**: In-game messages (200 characters)
- **Structured**: Map ID, direction, HP, levels, events (335 dimensions)

**Solution**: Separate encoders, then concatenate:

```python
class ObservationEncoder:
    def __init__(self):
        self.screen_encoder = CNN(...)
        self.visited_encoder = CNN(...)
        self.text_encoder = TransformerTextEncoder(...)
        self.state_encoder = MLP(...)
    
    def forward(self, obs):
        screen_feat = self.screen_encoder(obs['screen'])
        visited_feat = self.visited_encoder(obs['visited_mask'])
        text_feat = self.text_encoder(obs['text'])
        state_feat = self.state_encoder(obs['state'])
        
        return concat([screen_feat, visited_feat, text_feat, state_feat])
```

**Normalization**:
- Screen/visited: Divide by 255, subtract 0.5
- Text: No normalization (indices for embedding lookup)
- State: Min-max normalization based on known ranges

## Actor-Critic Training

### Imagination-Based Learning

**Key Idea**: Train policy on imagined trajectories instead of real experience

**Advantages**:
1. **Sample Efficiency**: One real experience → many imagined trajectories
2. **Credit Assignment**: Can look ahead arbitrary timesteps
3. **Exploration**: Can imagine "what if" scenarios

**Process**:
1. Sample initial states from replay buffer (real experience)
2. Roll out policy in imagination (using world model)
3. Compute λ-returns from imagined rewards and values
4. Update actor to maximize returns
5. Update critic to predict returns

### Critic Ensemble

**Chosen Approach**: Multiple critic networks (ensemble)

**Rationale**:
1. **Overestimation Bias**: Single critic tends to overestimate values
2. **Uncertainty**: Ensemble provides estimate of epistemic uncertainty
3. **Stability**: Averaging reduces variance

**Implementation**:
```python
class DreamerV3CriticEnsemble:
    def __init__(self, num_critics=2):
        self.critics = [DreamerV3Critic() for _ in range(num_critics)]
    
    def forward(self, latent, reduce='min'):
        values = [critic(latent) for critic in self.critics]
        if reduce == 'min':
            return torch.min(torch.stack(values), dim=0)
        elif reduce == 'mean':
            return torch.mean(torch.stack(values), dim=0)
        return values
```

**Target Network**:
- EMA (Exponential Moving Average) update: `θ' = τ * θ + (1-τ) * θ'`
- Default τ = 0.01 (slow update for stability)

### Symlog Transform

**Problem**: Value targets can have large magnitude and high variance

**Solution**: Symlog (symmetric logarithm) transformation

```python
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
```

**Properties**:
- Linear near zero: `symlog(x) ≈ x` for small x
- Logarithmic for large values: `symlog(x) ≈ sign(x) * log(|x|)` for large x
- Symmetric: `symlog(-x) = -symlog(x)`

**Benefits**:
- Reduces impact of outliers
- Makes learning more stable
- Better gradient flow

## Replay Buffer

### Sequence-Based Sampling

**Challenge**: RNNs require sequences, not individual transitions

**Solution**: Store and sample fixed-length sequences

```python
class SequenceReplayBuffer:
    def __init__(self, capacity, sequence_length, ...):
        self.capacity = capacity
        self.sequence_length = sequence_length
        # Pre-allocate arrays
        self.observations = {...}  # Dict of numpy arrays
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.dones = np.zeros(capacity, dtype=bool)
```

**Sampling Strategy**:
1. Choose random start indices
2. Extract sequences of length `sequence_length`
3. Mask out episode boundaries (don't cross episodes)

**Multi-Environment Support**:
- Separate buffer for each environment
- Maintains episode structure per environment
- Samples from all buffers proportionally

### Memory Management

**Optimization**: Pre-allocated numpy arrays

**Rationale**:
- Avoid dynamic allocation overhead
- Better cache locality
- Faster sampling (continuous memory)

**Implementation**:
```python
# Pre-allocate
self.observations['screen'] = np.zeros(
    (capacity, *screen_shape),
    dtype=np.uint8  # Use uint8 for images (4x smaller than float32)
)

# Add experience
self.observations['screen'][self.ptr] = new_observation

# Sample sequences
start = random.randint(0, self.size - self.sequence_length)
sequence = self.observations['screen'][start:start+self.sequence_length]
```

**Memory Usage**:
- 2M capacity with 72x80x1 screens ≈ 11 GB (uint8)
- Would be 44 GB with float32!

## Training Infrastructure

### Hybrid Mode: DreamerV3 + PPO

**Motivation**: Leverage existing PufferLib infrastructure

**Architecture**:
```python
class DreamerPPOPolicy(nn.Module):
    def __init__(self, world_model, freeze_world_model=False):
        self.representation = DreamerRepresentationEncoder(world_model)
        self.actor = nn.Sequential(...)  # PPO actor head
        self.value_fn = nn.Sequential(...)  # PPO critic head
    
    def forward(self, obs):
        latent = self.representation(obs)  # From world model
        action_logits = self.actor(latent)
        value = self.value_fn(latent)
        return action_logits, value
```

**Advantages**:
1. **Proven Infrastructure**: PufferLib's PPO is well-tested
2. **Flexibility**: Can pre-train world model, then use with PPO
3. **Debugging**: Easier to debug than full DreamerV3
4. **Performance**: Often more stable in practice

**Trade-offs**:
- Loses some benefits of imagination-based learning
- Still requires replay buffer for world model training
- More complex than pure PPO or pure DreamerV3

### Training Loop Design

**Pure DreamerV3**:
```
Loop:
  1. Collect experience (interact with environment)
  2. Train world model (on replay buffer)
  3. Train actor-critic (via imagination)
  4. Log metrics
  5. Save checkpoint
```

**Hybrid Mode**:
```
Loop:
  1. Collect experience (PPO-style rollouts)
  2. Train world model (on replay buffer)
  3. Train PPO (on rollouts, using world model representations)
  4. Log metrics
  5. Save checkpoint
```

### Automatic Mixed Precision (AMP)

**Implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=use_amp)

with autocast(enabled=use_amp):
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
nn.utils.clip_grad_norm_(parameters, grad_clip)
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 2-3x faster training on modern GPUs
- Reduced memory usage (float16 vs float32)
- Minimal impact on convergence (with gradient scaling)

## Testing Strategy

### Unit Tests

Each component has isolated tests:

1. **Text Extraction**: Character decoding, buffer reading
2. **RSSM**: State initialization, transitions, sequences
3. **Actor-Critic**: Action sampling, value prediction, training
4. **Replay Buffer**: Adding, sampling, wraparound

### Integration Tests

Test interactions between components:

1. **World Model**: Forward pass with multi-modal observations
2. **Imagination Rollout**: Actor + world model interaction
3. **End-to-End**: Full training iteration

### Test Design Principles

1. **Determinism**: Use fixed random seeds
2. **Small Scale**: Small models for fast tests
3. **Shape Checking**: Verify tensor dimensions
4. **Gradient Flow**: Check that gradients propagate

## Future Work

### Potential Improvements

1. **Pre-trained Text Encoder**:
   - Use BERT or similar for text understanding
   - Fine-tune on Pokemon-specific text
   - Challenge: Integration with world model training

2. **Hierarchical World Model**:
   - Multiple timescales (frame-level, action-level, goal-level)
   - Improves long-horizon planning
   - Challenge: Increased complexity

3. **Model-Based Planning**:
   - Use world model for explicit planning (e.g., MCTS)
   - Better for sparse reward environments
   - Challenge: Computational cost

4. **Multi-Task Learning**:
   - Train on multiple Pokemon games simultaneously
   - Shared world model, task-specific policies
   - Challenge: Task interference

### Known Limitations

1. **World Model Accuracy**:
   - Imperfect reconstruction of complex scenes
   - May struggle with rare events
   - Mitigation: Larger model capacity, more data

2. **Imagination Drift**:
   - Imagined trajectories diverge from reality
   - Compounds over long horizons
   - Mitigation: Shorter imagination horizon, better world model

3. **Computational Cost**:
   - Higher than model-free methods (PPO, DQN)
   - Requires GPU for reasonable training speed
   - Mitigation: Optimize implementation, use AMP

4. **Hyperparameter Sensitivity**:
   - Many hyperparameters to tune
   - Interactions between components
   - Mitigation: Use proven defaults, systematic tuning

## Lessons Learned

1. **Start Simple**: Get basic components working before adding complexity
2. **Test Early**: Unit tests catch bugs before integration
3. **Monitor Everything**: Log all metrics, visualize reconstructions
4. **Debug Systematically**: Isolate issues to specific components
5. **Use Proven Architectures**: Don't reinvent the wheel
6. **Document Decisions**: Future you will thank present you

## References and Resources

### Papers

1. [DreamerV3: Mastering Atari with Discrete World Models](https://arxiv.org/abs/2301.04104)
2. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) (DreamerV1)
3. [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193) (DreamerV2)

### Code References

1. [Official DreamerV3 Implementation](https://github.com/danijar/dreamerv3)
2. [PufferLib](https://github.com/PufferAI/PufferLib)
3. [Original Pokemon Red Experiment](https://github.com/thatguy11325/pokemonred_puffer)

### Tools

1. [PyTorch](https://pytorch.org/)
2. [Weights & Biases](https://wandb.ai/)
3. [PyBoy](https://github.com/Baekalfen/PyBoy)

## Acknowledgments

- DreamerV3 authors for the algorithm and paper
- PufferLib team for the RL infrastructure
- Original pokemonred_puffer contributors
- Pokemon Red disassembly community for memory addresses

---

*Last Updated: November 2025*

