"""
DreamerV3 World Model implementation.

This module implements the world model from DreamerV3, including:
- RSSM (Recurrent State-Space Model) for learning latent dynamics
- Observation encoder and decoder
- Reward and continue predictors
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from pokemonred_puffer.models.dreamer.utils import (
    CategoricalDistribution,
    DenseLayer,
    MLP,
    symlog,
    symexp,
)


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) from DreamerV3.
    
    The RSSM consists of:
    - Deterministic state h_t (from GRU): h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    - Stochastic state z_t (categorical): z_t ~ p(z_t | h_t)
    - Posterior: z_t ~ q(z_t | h_t, embedded_obs_t)
    """
    
    def __init__(
        self,
        action_dim: int,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 512,
        gru_layers: int = 1,
        unimix: float = 0.01,
    ):
        """
        Initialize RSSM.
        
        Args:
            action_dim: Dimension of action space
            deter_dim: Dimension of deterministic state
            stoch_dim: Number of categorical variables in stochastic state
            stoch_classes: Number of classes per categorical variable
            hidden_dim: Dimension of hidden layers
            gru_layers: Number of GRU layers
            unimix: Uniform mixture weight for categorical distributions
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.hidden_dim = hidden_dim
        self.unimix = unimix
        
        # Total stochastic dimension (flattened)
        self.stoch_total = stoch_dim * stoch_classes
        
        # GRU for deterministic state: h_t = f(h_{t-1}, [z_{t-1}, a_{t-1}])
        self.gru = nn.GRU(
            input_size=self.stoch_total + action_dim,
            hidden_size=deter_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        
        # Prior network: p(z_t | h_t)
        self.prior_mlp = MLP(
            input_dim=deter_dim,
            hidden_dims=[hidden_dim],
            output_dim=self.stoch_total,
            activation='silu',
        )
        
        # Posterior network: q(z_t | h_t, obs_embedding)
        self.posterior_mlp = MLP(
            input_dim=deter_dim + hidden_dim,  # h_t + embedded_obs
            hidden_dims=[hidden_dim],
            output_dim=self.stoch_total,
            activation='silu',
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Get initial state for RSSM.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Dictionary with 'deter' (deterministic state) and 'stoch' (stochastic state)
        """
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        
        return {'deter': deter, 'stoch': stoch}
    
    def get_stoch_dist(self, logits: torch.Tensor) -> CategoricalDistribution:
        """
        Get categorical distribution for stochastic state.
        
        Args:
            logits: Logits for categorical distribution, shape (batch, stoch_total)
            
        Returns:
            Categorical distribution
        """
        # Reshape to (batch, stoch_dim, stoch_classes)
        logits = logits.reshape(-1, self.stoch_dim, self.stoch_classes)
        return CategoricalDistribution(logits=logits, unimix=self.unimix)
    
    def sample_stoch(self, dist: CategoricalDistribution) -> torch.Tensor:
        """
        Sample from stochastic state distribution.
        
        Args:
            dist: Categorical distribution
            
        Returns:
            Sampled stochastic state, shape (batch, stoch_dim, stoch_classes)
        """
        # Sample one-hot vectors for each categorical variable
        samples = dist.sample()  # (batch, stoch_dim)
        stoch = F.one_hot(samples, self.stoch_classes).float()
        return stoch
    
    def flatten_stoch(self, stoch: torch.Tensor) -> torch.Tensor:
        """
        Flatten stochastic state from (..., stoch_dim, stoch_classes) to (..., stoch_total).
        Preserves all leading dimensions (batch, time, etc.).
        """
        # Get shape, preserving all leading dimensions
        leading_dims = stoch.shape[:-2]  # All dimensions except last two
        return stoch.reshape(*leading_dims, self.stoch_total)
    
    def get_latent(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get full latent state by concatenating deterministic and stochastic parts.
        
        Args:
            state: Dictionary with 'deter' and 'stoch'
            
        Returns:
            Concatenated latent state, shape (batch, deter_dim + stoch_total)
        """
        deter = state['deter']
        stoch_flat = self.flatten_stoch(state['stoch'])
        return torch.cat([deter, stoch_flat], dim=-1)
    
    def imagine_step(
        self, 
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], CategoricalDistribution]:
        """
        Imagine one step forward using the prior.
        
        Args:
            prev_state: Previous state with 'deter' and 'stoch'
            action: Action taken, shape (batch, action_dim)
            
        Returns:
            Tuple of (next_state, prior_dist)
        """
        # Flatten previous stochastic state
        prev_stoch_flat = self.flatten_stoch(prev_state['stoch'])
        
        # Concatenate previous stochastic state and action
        gru_input = torch.cat([prev_stoch_flat, action], dim=-1)
        gru_input = gru_input.unsqueeze(1)  # Add time dimension
        
        # Update deterministic state with GRU
        deter_h = prev_state['deter'].unsqueeze(0)  # (1, batch, deter_dim) for GRU
        new_deter_h, _ = self.gru(gru_input, deter_h)
        new_deter = new_deter_h.squeeze(0)  # Remove layer dimension
        new_deter = new_deter.squeeze(1)  # Remove time dimension
        
        # Compute prior distribution p(z_t | h_t)
        prior_logits = self.prior_mlp(new_deter)
        prior_dist = self.get_stoch_dist(prior_logits)
        
        # Sample stochastic state from prior
        new_stoch = self.sample_stoch(prior_dist)
        
        new_state = {'deter': new_deter, 'stoch': new_stoch}
        
        return new_state, prior_dist
    
    def observe_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        obs_embedding: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], CategoricalDistribution, CategoricalDistribution]:
        """
        Observe one step, computing both prior and posterior.
        
        Args:
            prev_state: Previous state with 'deter' and 'stoch'
            action: Action taken, shape (batch, action_dim)
            obs_embedding: Embedded observation, shape (batch, embedding_dim)
            
        Returns:
            Tuple of (next_state, prior_dist, posterior_dist)
        """
        # Flatten previous stochastic state
        prev_stoch_flat = self.flatten_stoch(prev_state['stoch'])
        
        # Concatenate previous stochastic state and action
        gru_input = torch.cat([prev_stoch_flat, action], dim=-1)
        gru_input = gru_input.unsqueeze(1)  # Add time dimension
        
        # Update deterministic state with GRU
        deter_h = prev_state['deter'].unsqueeze(0)  # (1, batch, deter_dim) for GRU
        new_deter_h, _ = self.gru(gru_input, deter_h)
        new_deter = new_deter_h.squeeze(0)  # Remove layer dimension
        new_deter = new_deter.squeeze(1)  # Remove time dimension
        
        # Compute prior distribution p(z_t | h_t)
        prior_logits = self.prior_mlp(new_deter)
        prior_dist = self.get_stoch_dist(prior_logits)
        
        # Compute posterior distribution q(z_t | h_t, obs_embedding)
        posterior_input = torch.cat([new_deter, obs_embedding], dim=-1)
        posterior_logits = self.posterior_mlp(posterior_input)
        posterior_dist = self.get_stoch_dist(posterior_logits)
        
        # Sample stochastic state from posterior
        new_stoch = self.sample_stoch(posterior_dist)
        
        new_state = {'deter': new_deter, 'stoch': new_stoch}
        
        return new_state, prior_dist, posterior_dist
    
    def observe_sequence(
        self,
        obs_embeddings: torch.Tensor,
        actions: torch.Tensor,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], list, list]:
        """
        Observe a sequence of observations and actions.
        
        Args:
            obs_embeddings: Sequence of embedded observations, shape (batch, time, embed_dim)
            actions: Sequence of actions, shape (batch, time, action_dim)
            initial_state: Initial state (if None, uses zero state)
            
        Returns:
            Tuple of:
            - states: Dictionary with 'deter' and 'stoch', each (batch, time, ...)
            - prior_dists: List of prior distributions
            - posterior_dists: List of posterior distributions
        """
        batch_size, seq_len, _ = obs_embeddings.shape
        device = obs_embeddings.device
        
        if initial_state is None:
            state = self.initial_state(batch_size, device)
        else:
            state = initial_state
        
        # Lists to store outputs
        deter_states = []
        stoch_states = []
        prior_dists = []
        posterior_dists = []
        
        # Process sequence
        for t in range(seq_len):
            obs_emb_t = obs_embeddings[:, t]
            action_t = actions[:, t]
            
            state, prior_dist, posterior_dist = self.observe_step(
                state, action_t, obs_emb_t
            )
            
            deter_states.append(state['deter'])
            stoch_states.append(state['stoch'])
            prior_dists.append(prior_dist)
            posterior_dists.append(posterior_dist)
        
        # Stack states
        states = {
            'deter': torch.stack(deter_states, dim=1),  # (batch, time, deter_dim)
            'stoch': torch.stack(stoch_states, dim=1),  # (batch, time, stoch_dim, stoch_classes)
        }
        
        return states, prior_dists, posterior_dists
    
    def imagine_sequence(
        self,
        initial_state: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], list]:
        """
        Imagine a sequence of states given actions (for planning/imagination).
        
        Args:
            initial_state: Initial state
            actions: Sequence of actions, shape (batch, time, action_dim)
            
        Returns:
            Tuple of (states, prior_dists)
        """
        batch_size, seq_len, _ = actions.shape
        
        state = initial_state
        
        deter_states = []
        stoch_states = []
        prior_dists = []
        
        for t in range(seq_len):
            action_t = actions[:, t]
            
            state, prior_dist = self.imagine_step(state, action_t)
            
            deter_states.append(state['deter'])
            stoch_states.append(state['stoch'])
            prior_dists.append(prior_dist)
        
        states = {
            'deter': torch.stack(deter_states, dim=1),
            'stoch': torch.stack(stoch_states, dim=1),
        }
        
        return states, prior_dists


class ObservationEncoder(nn.Module):
    """
    Encoder for Pokemon Red observations.
    
    Encodes screen, text, and other game state into a unified embedding.
    """
    
    def __init__(
        self,
        screen_shape: Tuple[int, int, int],
        text_encoder: nn.Module,
        state_dim: int,
        output_dim: int = 512,
        cnn_depth: int = 32,
        two_bit: bool = True,
    ):
        """
        Initialize observation encoder.
        
        Args:
            screen_shape: Shape of screen observation (H, W, C) - compressed if two_bit=True
            text_encoder: Text encoder module
            state_dim: Dimension of other state features (items, party, etc.)
            output_dim: Output embedding dimension
            cnn_depth: Base depth for CNN
            two_bit: Whether observations use two-bit compression
        """
        super().__init__()
        
        self.screen_shape = screen_shape
        self.text_encoder = text_encoder
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.two_bit = two_bit
        
        # Calculate restored shape if two_bit compression is used
        if self.two_bit:
            # Restore width by multiplying by 4, keep height same
            self.restored_shape = (screen_shape[0], screen_shape[1] * 4, screen_shape[2])
        else:
            self.restored_shape = screen_shape
        
        # Setup two_bit unpacking buffers (similar to multi_convolutional policy)
        if self.two_bit:
            self.register_buffer(
                'screen_buckets',
                torch.tensor([0, 85, 153, 255], dtype=torch.float32) / 255.0
            )
            self.register_buffer(
                'linear_buckets',
                torch.tensor([0, 85, 153, 255], dtype=torch.float32) / 255.0
            )
            self.register_buffer(
                'unpack_mask',
                torch.tensor([[192, 48, 12, 3]], dtype=torch.uint8)
            )
            self.register_buffer(
                'unpack_shift',
                torch.tensor([[6, 4, 2, 0]], dtype=torch.uint8)
            )
        
        # Screen CNN encoder (similar to existing policy but for DreamerV3)
        # Use restored shape for CNN input
        self.screen_cnn = nn.Sequential(
            nn.Conv2d(self.restored_shape[2] * 2, cnn_depth, 8, stride=2),  # *2 for screen + visited_mask
            nn.SiLU(),
            nn.Conv2d(cnn_depth, cnn_depth * 2, 4, stride=2),
            nn.SiLU(),
            nn.Conv2d(cnn_depth * 2, cnn_depth * 4, 3, stride=2),
            nn.SiLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output dimension
        with torch.no_grad():
            dummy_screen = torch.zeros(1, self.restored_shape[2] * 2, 
                                      self.restored_shape[0], self.restored_shape[1])
            cnn_out_dim = self.screen_cnn(dummy_screen).shape[1]
        
        # State MLP encoder
        self.state_mlp = MLP(
            input_dim=state_dim,
            hidden_dims=[256, 256],
            output_dim=256,
            activation='silu',
        )
        
        # Combine all embeddings
        total_dim = cnn_out_dim + text_encoder.output_dim + 256
        self.fusion_mlp = MLP(
            input_dim=total_dim,
            hidden_dims=[output_dim],
            output_dim=output_dim,
            activation='silu',
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observations to embedding.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Observation embedding, shape (batch, output_dim)
        """
        # Process screen (concatenate screen and visited_mask)
        screen = observations['screen']
        visited_mask = observations['visited_mask']
        
        # Unpack two_bit compression if needed
        if self.two_bit:
            batch_size = screen.shape[0]
            restored_shape = (batch_size, self.restored_shape[0], self.restored_shape[1], self.restored_shape[2])
            
            # Unpack screen
            screen = torch.index_select(
                self.screen_buckets,
                0,
                ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
            ).reshape(restored_shape)
            
            # Unpack visited_mask
            visited_mask = torch.index_select(
                self.linear_buckets,
                0,
                ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
            ).reshape(restored_shape)
        else:
            # Normalize to [0, 1]
            screen = screen.float() / 255.0
            visited_mask = visited_mask.float() / 255.0
        
        # Permute from (batch, H, W, C) to (batch, C, H, W)
        screen = screen.permute(0, 3, 1, 2)
        visited_mask = visited_mask.permute(0, 3, 1, 2)
        screen_input = torch.cat([screen, visited_mask], dim=1)
        
        screen_emb = self.screen_cnn(screen_input)
        
        # Process text
        text_ids = observations['text']
        text_emb = self.text_encoder(text_ids)
        
        # Process other state features (concatenate relevant observations)
        state_features = []
        for key in ['direction', 'map_id', 'battle_type', 'level', 'hp', 'events']:
            if key in observations:
                feat = observations[key].float()
                if feat.dim() == 1:
                    feat = feat.unsqueeze(-1)
                state_features.append(feat)
        
        state_input = torch.cat(state_features, dim=-1)
        state_emb = self.state_mlp(state_input)
        
        # Fuse all embeddings
        combined = torch.cat([screen_emb, text_emb, state_emb], dim=-1)
        embedding = self.fusion_mlp(combined)
        
        return embedding


class ObservationDecoder(nn.Module):
    """
    Decoder for reconstructing observations from latent state.
    
    In DreamerV3, we typically only reconstruct the screen for efficiency.
    """
    
    def __init__(
        self,
        latent_dim: int,
        screen_shape: Tuple[int, int, int],
        cnn_depth: int = 32,
    ):
        """
        Initialize observation decoder.
        
        Args:
            latent_dim: Dimension of latent state
            screen_shape: Shape of screen to reconstruct (H, W, C)
            cnn_depth: Base depth for CNN
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.screen_shape = screen_shape
        
        # Project latent to spatial features
        self.proj = nn.Linear(latent_dim, cnn_depth * 4 * 9 * 10)  # Approximate spatial size
        
        # Transposed CNN decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (cnn_depth * 4, 9, 10)),
            nn.ConvTranspose2d(cnn_depth * 4, cnn_depth * 2, 3, stride=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(cnn_depth * 2, cnn_depth, 4, stride=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(cnn_depth, screen_shape[2], 8, stride=2, output_padding=1),
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to screen observation.
        
        Args:
            latent: Latent state, shape (batch, latent_dim)
            
        Returns:
            Reconstructed screen, shape (batch, C, H, W)
        """
        x = self.proj(latent)
        x = self.decoder(x)
        
        # Resize to exact screen shape if needed
        if x.shape[2:] != (self.screen_shape[0], self.screen_shape[1]):
            x = F.interpolate(x, size=(self.screen_shape[0], self.screen_shape[1]), mode='bilinear')
        
        return x


class RewardPredictor(nn.Module):
    """Predict rewards from latent state."""
    
    def __init__(self, latent_dim: int, hidden_dims: list = [512, 512]):
        super().__init__()
        self.mlp = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='silu',
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward in symlog space."""
        return self.mlp(latent).squeeze(-1)


class ContinuePredictor(nn.Module):
    """Predict episode continuation (1 - done) from latent state."""
    
    def __init__(self, latent_dim: int, hidden_dims: list = [512, 512]):
        super().__init__()
        self.mlp = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='silu',
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict continue probability."""
        logits = self.mlp(latent).squeeze(-1)
        return torch.sigmoid(logits)


class DreamerV3WorldModel(nn.Module):
    """
    Complete DreamerV3 World Model.
    
    Combines RSSM, encoder, decoder, reward predictor, and continue predictor.
    """
    
    def __init__(
        self,
        action_dim: int,
        screen_shape: Tuple[int, int, int],
        text_encoder: nn.Module,
        state_dim: int,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 512,
        encoder_dim: int = 512,
        two_bit: bool = True,
    ):
        """
        Initialize world model.
        
        Args:
            action_dim: Dimension of action space
            screen_shape: Shape of screen observation (compressed if two_bit=True)
            text_encoder: Text encoder module
            state_dim: Dimension of other state features
            deter_dim: Dimension of deterministic RSSM state
            stoch_dim: Number of categorical variables in stochastic state
            stoch_classes: Number of classes per categorical variable
            hidden_dim: Dimension of hidden layers
            encoder_dim: Dimension of observation encoder output
            two_bit: Whether observations use two-bit compression
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.screen_shape = screen_shape
        self.two_bit = two_bit
        
        # RSSM
        self.rssm = RSSM(
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            stoch_classes=stoch_classes,
            hidden_dim=hidden_dim,
        )
        
        # Encoder
        self.encoder = ObservationEncoder(
            screen_shape=screen_shape,
            text_encoder=text_encoder,
            state_dim=state_dim,
            output_dim=encoder_dim,
            two_bit=two_bit,
        )
        
        # Decoder
        self.decoder = ObservationDecoder(
            latent_dim=deter_dim + stoch_dim * stoch_classes,
            screen_shape=screen_shape,
        )
        
        # Reward and continue predictors
        self.reward_predictor = RewardPredictor(
            latent_dim=deter_dim + stoch_dim * stoch_classes,
        )
        self.continue_predictor = ContinuePredictor(
            latent_dim=deter_dim + stoch_dim * stoch_classes,
        )
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, any]:
        """
        Forward pass through world model (for training).
        
        Args:
            observations: Dictionary of observations, each (batch, time, ...)
            actions: Actions, shape (batch, time, action_dim)
            initial_state: Initial RSSM state
            
        Returns:
            Dictionary with predictions and distributions
        """
        # Encode observations
        batch_size, seq_len = actions.shape[:2]
        
        # Flatten time dimension for encoding
        obs_flat = {}
        for key, val in observations.items():
            if val.dim() > 2:
                obs_flat[key] = val.reshape(batch_size * seq_len, *val.shape[2:])
            else:
                obs_flat[key] = val.reshape(batch_size * seq_len, *val.shape[2:])
        
        obs_embeddings_flat = self.encoder(obs_flat)
        obs_embeddings = obs_embeddings_flat.reshape(batch_size, seq_len, -1)
        
        # Run RSSM to get latent states
        states, prior_dists, posterior_dists = self.rssm.observe_sequence(
            obs_embeddings, actions, initial_state
        )
        
        # Get latent representations
        latents = self.rssm.get_latent(states)  # (batch, time, latent_dim)
        
        # Flatten for predictions
        latents_flat = latents.reshape(batch_size * seq_len, -1)
        
        # Decode observations
        screen_recon = self.decoder(latents_flat)
        screen_recon = screen_recon.reshape(batch_size, seq_len, *screen_recon.shape[1:])
        
        # Predict rewards and continues (in symlog space)
        reward_pred = self.reward_predictor(latents_flat)
        reward_pred = reward_pred.reshape(batch_size, seq_len)
        
        continue_pred = self.continue_predictor(latents_flat)
        continue_pred = continue_pred.reshape(batch_size, seq_len)
        
        return {
            'states': states,
            'latents': latents,
            'prior_dists': prior_dists,
            'posterior_dists': posterior_dists,
            'screen_recon': screen_recon,
            'reward_pred': reward_pred,
            'continue_pred': continue_pred,
            'obs_embeddings': obs_embeddings,
        }

