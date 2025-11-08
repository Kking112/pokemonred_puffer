"""
World Model Trainer for DreamerV3.

Handles training of the world model including RSSM, encoder, decoder,
reward predictor, and continue predictor.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pokemonred_puffer.models.dreamer.world_model import DreamerV3WorldModel
from pokemonred_puffer.models.dreamer.replay_buffer import MultiEnvReplayBuffer
from pokemonred_puffer.models.dreamer.utils import symlog, symexp


class WorldModelTrainer:
    """
    Trainer for DreamerV3 world model.
    
    Implements the training procedure for all world model components including:
    - RSSM (prior and posterior)
    - Observation decoder
    - Reward predictor
    - Continue predictor
    """
    
    def __init__(
        self,
        world_model: DreamerV3WorldModel,
        learning_rate: float = 1e-4,
        grad_clip: float = 100.0,
        kl_weight: float = 1.0,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
        reconstruction_weight: float = 1.0,
        reward_weight: float = 1.0,
        continue_weight: float = 1.0,
        use_amp: bool = False,
    ):
        """
        Initialize world model trainer.
        
        Args:
            world_model: World model to train
            learning_rate: Learning rate
            grad_clip: Gradient clipping value
            kl_weight: Weight for KL divergence loss
            kl_balance: Balance between prior and posterior KL (DreamerV3 trick)
            free_nats: Free nats for KL loss (prevent posterior collapse)
            reconstruction_weight: Weight for reconstruction loss
            reward_weight: Weight for reward prediction loss
            continue_weight: Weight for continue prediction loss
            use_amp: Whether to use automatic mixed precision
        """
        self.world_model = world_model
        self.grad_clip = grad_clip
        self.kl_weight = kl_weight
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.reconstruction_weight = reconstruction_weight
        self.reward_weight = reward_weight
        self.continue_weight = continue_weight
        self.use_amp = use_amp
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            world_model.parameters(),
            lr=learning_rate,
            eps=1e-5
        )
        
        # AMP scaler for mixed precision training
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.train_step_count = 0
    
    def compute_kl_loss(
        self,
        prior_dists: list,
        posterior_dists: list,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between prior and posterior with free nats.
        
        Uses the DreamerV3 trick of balancing KL between forward and backward directions.
        
        Args:
            prior_dists: List of prior distributions
            posterior_dists: List of posterior distributions
            
        Returns:
            KL loss
        """
        kl_losses = []
        
        for prior_dist, posterior_dist in zip(prior_dists, posterior_dists):
            # Forward KL: KL(posterior || prior)
            # This measures how much information the posterior has that prior doesn't
            # Shape: (batch, stoch_dim)
            kl_forward = torch.distributions.kl_divergence(
                torch.distributions.Categorical(probs=posterior_dist.probs),
                torch.distributions.Categorical(probs=prior_dist.probs),
            )
            
            # Backward KL: KL(prior || posterior) 
            # This measures how much the prior could be improved
            kl_backward = torch.distributions.kl_divergence(
                torch.distributions.Categorical(probs=prior_dist.probs),
                torch.distributions.Categorical(probs=posterior_dist.probs),
            )
            
            # Balance between forward and backward
            kl = self.kl_balance * kl_forward + (1 - self.kl_balance) * kl_backward
            
            # Sum over stochastic dimensions
            kl = kl.sum(dim=-1)  # (batch,)
            
            # Apply free nats (don't penalize KL below this threshold)
            kl = torch.maximum(kl, torch.tensor(self.free_nats, device=kl.device))
            
            kl_losses.append(kl)
        
        # Stack and mean over time and batch
        kl_loss = torch.stack(kl_losses, dim=1).mean()
        
        return kl_loss
    
    def compute_reconstruction_loss(
        self,
        screen_recon: torch.Tensor,
        screen_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for screen observations.
        
        Args:
            screen_recon: Reconstructed screen, shape (batch, time, C, H, W)
            screen_target: Target screen, shape (batch, time, C, H, W)
            
        Returns:
            Reconstruction loss
        """
        # Normalize targets to [0, 1]
        screen_target = screen_target.float() / 255.0
        
        # Permute from (batch, time, H, W, C) to (batch, time, C, H, W) if needed
        if screen_target.dim() == 5 and screen_target.shape[-1] <= 4:
            screen_target = screen_target.permute(0, 1, 4, 2, 3)
        
        # MSE loss
        recon_loss = F.mse_loss(screen_recon, screen_target)
        
        return recon_loss
    
    def compute_reward_loss(
        self,
        reward_pred: torch.Tensor,
        reward_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reward prediction loss (in symlog space).
        
        Args:
            reward_pred: Predicted rewards (symlog space), shape (batch, time)
            reward_target: Target rewards, shape (batch, time)
            
        Returns:
            Reward loss
        """
        # Convert target to symlog space
        reward_target_symlog = symlog(reward_target)
        
        # MSE loss in symlog space
        reward_loss = F.mse_loss(reward_pred, reward_target_symlog)
        
        return reward_loss
    
    def compute_continue_loss(
        self,
        continue_pred: torch.Tensor,
        done_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute continue prediction loss.
        
        Args:
            continue_pred: Predicted continue probabilities, shape (batch, time)
            done_target: Target done flags, shape (batch, time)
            
        Returns:
            Continue loss
        """
        # Continue = 1 - done
        continue_target = (1.0 - done_target.float())
        
        # Binary cross-entropy loss
        continue_loss = F.binary_cross_entropy(
            continue_pred,
            continue_target,
        )
        
        return continue_loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step for world model.
        
        Args:
            batch: Batch of data from replay buffer containing:
                - observations: Dict of observation sequences
                - actions: Action sequences
                - rewards: Reward sequences
                - dones: Done flags
                
        Returns:
            Dictionary of training metrics
        """
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        
        self.optimizer.zero_grad()
        
        # Mixed precision context
        amp_context = torch.cuda.amp.autocast() if self.use_amp else torch.no_grad
        
        with amp_context():
            # Forward pass through world model
            outputs = self.world_model(observations, actions)
            
            # Extract outputs
            prior_dists = outputs['prior_dists']
            posterior_dists = outputs['posterior_dists']
            screen_recon = outputs['screen_recon']
            reward_pred = outputs['reward_pred']
            continue_pred = outputs['continue_pred']
            
            # Compute losses
            kl_loss = self.compute_kl_loss(prior_dists, posterior_dists)
            
            # Screen reconstruction loss
            screen_target = observations['screen']
            recon_loss = self.compute_reconstruction_loss(screen_recon, screen_target)
            
            # Reward prediction loss
            reward_loss = self.compute_reward_loss(reward_pred, rewards)
            
            # Continue prediction loss
            continue_loss = self.compute_continue_loss(continue_pred, dones)
            
            # Total loss
            total_loss = (
                self.kl_weight * kl_loss +
                self.reconstruction_weight * recon_loss +
                self.reward_weight * reward_loss +
                self.continue_weight * continue_loss
            )
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(), self.grad_clip
            )
            self.optimizer.step()
        
        self.train_step_count += 1
        
        # Return metrics
        metrics = {
            'total_loss': total_loss.item(),
            'kl_loss': kl_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item(),
            'grad_norm': grad_norm.item(),
        }
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """
        Save checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'world_model_state_dict': self.world_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step_count': self.train_step_count,
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path)
        
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step_count = checkpoint.get('train_step_count', 0)
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


class ImaginationRollout:
    """
    Helper class for performing imagination rollouts with the world model.
    
    Used for training actor-critic via imagination.
    """
    
    def __init__(
        self,
        world_model: DreamerV3WorldModel,
        actor: nn.Module,
        horizon: int = 15,
    ):
        """
        Initialize imagination rollout.
        
        Args:
            world_model: Trained world model
            actor: Actor network
            horizon: Imagination horizon (number of steps to imagine)
        """
        self.world_model = world_model
        self.actor = actor
        self.horizon = horizon
    
    @torch.no_grad()
    def rollout(
        self,
        initial_states: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform imagination rollout from initial states.
        
        Args:
            initial_states: Initial RSSM states from real experience
            deterministic: If True, use deterministic actions
            
        Returns:
            Dictionary containing:
            - latents: Imagined latent states, (batch, horizon, latent_dim)
            - actions: Imagined actions, (batch, horizon, action_dim)
            - rewards: Predicted rewards, (batch, horizon)
            - continues: Predicted continues, (batch, horizon)
        """
        batch_size = initial_states['deter'].shape[0]
        device = initial_states['deter'].device
        
        # Storage for rollout
        latents = []
        actions_list = []
        rewards = []
        continues = []
        
        # Current state
        state = initial_states
        
        for t in range(self.horizon):
            # Get latent representation
            latent = self.world_model.rssm.get_latent(state)
            latents.append(latent)
            
            # Sample action from actor
            action, _ = self.actor(latent, deterministic=deterministic)
            actions_list.append(action)
            
            # Predict reward and continue
            reward_symlog = self.world_model.reward_predictor(latent)
            continue_prob = self.world_model.continue_predictor(latent)
            
            rewards.append(reward_symlog)
            continues.append(continue_prob)
            
            # Imagine next state
            state, _ = self.world_model.rssm.imagine_step(state, action)
        
        # Stack tensors
        latents = torch.stack(latents, dim=1)  # (batch, horizon, latent_dim)
        actions = torch.stack(actions_list, dim=1)  # (batch, horizon, action_dim)
        rewards = torch.stack(rewards, dim=1)  # (batch, horizon)
        continues = torch.stack(continues, dim=1)  # (batch, horizon)
        
        return {
            'latents': latents,
            'actions': actions,
            'rewards': rewards,
            'continues': continues,
        }

