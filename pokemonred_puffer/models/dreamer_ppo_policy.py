"""
Hybrid policy using DreamerV3 world model for representation learning
with PPO for policy optimization.

This allows leveraging PufferLib infrastructure while benefiting from
the learned latent representations of DreamerV3.
"""

import torch
import torch.nn as nn
import pufferlib.pytorch
import pufferlib.models

from pokemonred_puffer.models.text_encoder import TransformerTextEncoder
from pokemonred_puffer.models.dreamer.world_model import DreamerV3WorldModel


class DreamerRepresentationEncoder(nn.Module):
    """
    Uses DreamerV3 world model to encode observations into latent representations.
    
    This encoder can be frozen (using pre-trained world model) or jointly trained.
    """
    
    def __init__(
        self,
        world_model: DreamerV3WorldModel,
        freeze_world_model: bool = False,
    ):
        """
        Initialize representation encoder.
        
        Args:
            world_model: Pre-trained or jointly trained world model
            freeze_world_model: If True, freeze world model parameters
        """
        super().__init__()
        
        self.world_model = world_model
        self.rssm = world_model.rssm
        self.encoder = world_model.encoder
        
        # Freeze world model if requested
        if freeze_world_model:
            for param in self.world_model.parameters():
                param.requires_grad = False
        
        # State tracking for sequential processing
        self.reset_state()
    
    def reset_state(self):
        """Reset RSSM state (call at episode start)."""
        self.current_state = None
        self.last_action = None
    
    def forward(self, observations: dict, action: torch.Tensor = None) -> torch.Tensor:
        """
        Encode observations to latent representation.
        
        Args:
            observations: Dictionary of observations
            action: Previous action (optional, for recurrent state update)
            
        Returns:
            Latent representation, shape (batch, latent_dim)
        """
        batch_size = next(iter(observations.values())).shape[0]
        device = next(iter(observations.values())).device
        
        # Initialize state if needed
        if self.current_state is None:
            self.current_state = self.rssm.initial_state(batch_size, device)
        
        # Encode observation
        obs_embedding = self.encoder(observations)
        
        # Update state
        if action is not None and self.last_action is None:
            # First step, use zero action
            self.last_action = torch.zeros(batch_size, action.shape[-1], device=device)
        
        if action is not None:
            # Update with posterior (we have observations)
            self.current_state, _, _ = self.rssm.observe_step(
                self.current_state,
                self.last_action,
                obs_embedding,
            )
            self.last_action = action
        else:
            # Just use current state with zero action
            if self.last_action is None:
                self.last_action = torch.zeros(batch_size, self.rssm.action_dim, device=device)
        
        # Get latent representation
        latent = self.rssm.get_latent(self.current_state)
        
        return latent


class DreamerPPOPolicy(nn.Module):
    """
    PPO policy using DreamerV3 representations.
    
    Compatible with PufferLib's CleanRL PPO implementation.
    """
    
    def __init__(
        self,
        env,
        world_model: DreamerV3WorldModel,
        freeze_world_model: bool = False,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        """
        Initialize DreamerPPO policy.
        
        Args:
            env: Gym environment
            world_model: DreamerV3 world model for encoding
            freeze_world_model: Whether to freeze world model
            hidden_dim: Hidden dimension for policy/value heads
            num_layers: Number of layers in policy/value heads
        """
        super().__init__()
        
        self.num_actions = env.single_action_space.n
        
        # Representation encoder
        self.representation = DreamerRepresentationEncoder(
            world_model=world_model,
            freeze_world_model=freeze_world_model,
        )
        
        # Get latent dimension
        latent_dim = (
            world_model.rssm.deter_dim + 
            world_model.rssm.stoch_dim * world_model.rssm.stoch_classes
        )
        
        # Policy head (actor)
        actor_layers = []
        prev_dim = latent_dim
        for _ in range(num_layers):
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.LayerNorm(hidden_dim))
            actor_layers.append(nn.SiLU())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, self.num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # Value head (critic)
        critic_layers = []
        prev_dim = latent_dim
        for _ in range(num_layers):
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(nn.SiLU())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.value_fn = nn.Sequential(*critic_layers)
    
    def forward(self, observations, action=None):
        """
        Forward pass for PPO.
        
        Args:
            observations: Observation dictionary
            action: Action for evaluation (optional)
            
        Returns:
            Tuple of (action_logits, log_probs, entropy, value)
            or (action_logits, log_probs, entropy, value, None) for compatibility
        """
        # Ensure observations are proper tensors
        obs_dict = {}
        for key, val in observations.items():
            if not isinstance(val, torch.Tensor):
                val = torch.from_numpy(val)
            obs_dict[key] = val
        
        # Get latent representation
        latent = self.representation(obs_dict, action)
        
        # Get action logits
        action_logits = self.actor(latent)
        
        # Get value
        value = self.value_fn(latent)
        
        # Compute action distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            # Sample action
            action = dist.sample()
        
        # Compute log prob and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action_logits, log_prob, entropy, value


class DreamerPPORNN(pufferlib.models.LSTMWrapper):
    """
    Recurrent version of DreamerPPO policy.
    
    This wraps the policy with an LSTM, though the RSSM already provides
    recurrence. Use this if you want additional LSTM layers on top.
    """
    
    def __init__(
        self,
        env,
        policy,
        input_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 1,
    ):
        """
        Initialize recurrent DreamerPPO policy.
        
        Args:
            env: Gym environment
            policy: Base DreamerPPO policy
            input_size: Input size for LSTM
            hidden_size: Hidden size for LSTM
            num_layers: Number of LSTM layers
        """
        super().__init__(env, policy, input_size, hidden_size, num_layers)


def create_dreamer_ppo_policy(
    env,
    config,
    world_model: DreamerV3WorldModel = None,
    freeze_world_model: bool = False,
):
    """
    Factory function to create DreamerPPO policy.
    
    Args:
        env: Gym environment
        config: Configuration dict
        world_model: Pre-trained world model (if None, creates new one)
        freeze_world_model: Whether to freeze world model
        
    Returns:
        Policy wrapped with PufferLib compatibility
    """
    # Create world model if not provided
    if world_model is None:
        from pokemonred_puffer.models.text_encoder import TransformerTextEncoder
        import numpy as np
        
        # Build text encoder
        text_encoder = TransformerTextEncoder(
            vocab_size=256,
            embed_dim=config.get('text_embed_dim', 128),
            num_heads=config.get('text_num_heads', 4),
            num_layers=config.get('text_num_layers', 2),
            output_dim=config.get('text_output_dim', 128),
        )
        
        # Get observation info
        obs_space = env.env.observation_space if hasattr(env, 'env') else env.observation_space
        screen_shape = obs_space.spaces['screen'].shape
        
        # Calculate state dimension
        state_dim = sum([
            np.prod(space.shape) for name, space in obs_space.spaces.items()
            if name not in ['screen', 'visited_mask', 'global_map', 'text']
        ])
        
        # Create world model
        action_dim = env.single_action_space.n
        world_model = DreamerV3WorldModel(
            action_dim=action_dim,
            screen_shape=screen_shape,
            text_encoder=text_encoder,
            state_dim=int(state_dim),
            deter_dim=config.get('deter_dim', 512),
            stoch_dim=config.get('stoch_dim', 32),
            stoch_classes=config.get('stoch_classes', 32),
        )
    
    # Create policy
    policy = DreamerPPOPolicy(
        env=env,
        world_model=world_model,
        freeze_world_model=freeze_world_model,
        hidden_dim=config.get('policy_hidden_dim', 512),
        num_layers=config.get('policy_num_layers', 2),
    )
    
    # Wrap with PufferLib if needed
    if config.get('use_rnn', False):
        policy = DreamerPPORNN(
            env=env,
            policy=policy,
            hidden_size=config.get('rnn_hidden_size', 512),
            num_layers=config.get('rnn_num_layers', 1),
        )
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)
    
    return policy

