"""
DreamerV3 Actor and Critic networks.

These networks operate in the learned latent space and are trained via
imagination rollouts from the world model.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as td

from pokemonred_puffer.models.dreamer.utils import (
    MLP,
    compute_lambda_returns,
    symlog,
    symexp,
    two_hot_encode,
    two_hot_decode,
)


class DreamerV3Actor(nn.Module):
    """
    Actor network for DreamerV3.
    
    Outputs a policy over actions given the latent state.
    For discrete actions, uses a categorical distribution.
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list = [512, 512],
        discrete: bool = True,
        num_action_bins: int = 255,  # For continuous actions with two-hot encoding
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        """
        Initialize actor network.
        
        Args:
            latent_dim: Dimension of latent state
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            discrete: Whether actions are discrete
            num_action_bins: Number of bins for two-hot encoding (continuous actions)
            action_low: Lower bound for continuous actions
            action_high: Upper bound for continuous actions
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.num_action_bins = num_action_bins
        self.action_low = action_low
        self.action_high = action_high
        
        if discrete:
            # Output logits for categorical distribution
            self.mlp = MLP(
                input_dim=latent_dim,
                hidden_dims=hidden_dims,
                output_dim=action_dim,
                activation='silu',
            )
        else:
            # Output two-hot encoding for continuous actions
            self.mlp = MLP(
                input_dim=latent_dim,
                hidden_dims=hidden_dims,
                output_dim=action_dim * num_action_bins,
                activation='silu',
            )
    
    def forward(self, latent: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor.
        
        Args:
            latent: Latent state, shape (batch, latent_dim)
            deterministic: If True, return mode instead of sampling
            
        Returns:
            Tuple of (actions, log_probs)
        """
        if self.discrete:
            logits = self.mlp(latent)
            dist = td.Categorical(logits=logits)
            
            if deterministic:
                actions = dist.probs.argmax(dim=-1)
            else:
                actions = dist.sample()
            
            log_probs = dist.log_prob(actions)
            
            return actions, log_probs
        else:
            # Two-hot encoding for continuous actions
            logits = self.mlp(latent)
            logits = logits.reshape(-1, self.action_dim, self.num_action_bins)
            
            # Categorical distribution for each action dimension
            actions = []
            log_probs = []
            
            for i in range(self.action_dim):
                dist = td.Categorical(logits=logits[:, i])
                
                if deterministic:
                    action_idx = dist.probs.argmax(dim=-1)
                else:
                    action_idx = dist.sample()
                
                # Convert bin index to continuous value
                action = (action_idx.float() / (self.num_action_bins - 1)) * \
                        (self.action_high - self.action_low) + self.action_low
                
                actions.append(action)
                log_probs.append(dist.log_prob(action_idx))
            
            actions = torch.stack(actions, dim=-1)
            log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
            
            return actions, log_probs
    
    def get_action_dist(self, latent: torch.Tensor):
        """
        Get action distribution without sampling.
        
        Args:
            latent: Latent state
            
        Returns:
            Action distribution
        """
        if self.discrete:
            logits = self.mlp(latent)
            return td.Categorical(logits=logits)
        else:
            logits = self.mlp(latent)
            logits = logits.reshape(-1, self.action_dim, self.num_action_bins)
            # Return list of distributions for each action dimension
            return [td.Categorical(logits=logits[:, i]) for i in range(self.action_dim)]
    
    def evaluate_actions(self, latent: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy of given actions.
        
        Args:
            latent: Latent state
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_probs, entropy)
        """
        if self.discrete:
            logits = self.mlp(latent)
            dist = td.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, entropy
        else:
            logits = self.mlp(latent)
            logits = logits.reshape(-1, self.action_dim, self.num_action_bins)
            
            log_probs_list = []
            entropy_list = []
            
            for i in range(self.action_dim):
                dist = td.Categorical(logits=logits[:, i])
                
                # Convert continuous action to bin index
                action_i = actions[:, i]
                action_idx = ((action_i - self.action_low) / (self.action_high - self.action_low) * \
                             (self.num_action_bins - 1)).long()
                action_idx = torch.clamp(action_idx, 0, self.num_action_bins - 1)
                
                log_probs_list.append(dist.log_prob(action_idx))
                entropy_list.append(dist.entropy())
            
            log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
            entropy = torch.stack(entropy_list, dim=-1).sum(dim=-1)
            
            return log_probs, entropy


class DreamerV3Critic(nn.Module):
    """
    Critic network for DreamerV3.
    
    Outputs value estimates in symlog space.
    DreamerV3 uses an ensemble of critics for stability.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [512, 512],
        num_bins: int = 255,
        value_low: float = -20.0,
        value_high: float = 20.0,
    ):
        """
        Initialize critic network.
        
        Args:
            latent_dim: Dimension of latent state
            hidden_dims: Hidden layer dimensions
            num_bins: Number of bins for two-hot value encoding
            value_low: Lower bound for value range (in symlog space)
            value_high: Upper bound for value range (in symlog space)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_bins = num_bins
        self.value_low = value_low
        self.value_high = value_high
        
        # Output two-hot encoding for value distribution
        self.mlp = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=num_bins,
            activation='silu',
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic.
        
        Args:
            latent: Latent state, shape (batch, latent_dim)
            
        Returns:
            Value estimate (in symlog space), shape (batch,)
        """
        logits = self.mlp(latent)
        probs = F.softmax(logits, dim=-1)
        
        # Decode two-hot to get value
        value_symlog = two_hot_decode(probs, self.num_bins, self.value_low, self.value_high)
        
        return value_symlog
    
    def get_value_dist(self, latent: torch.Tensor) -> td.Categorical:
        """
        Get value distribution.
        
        Args:
            latent: Latent state
            
        Returns:
            Categorical distribution over value bins
        """
        logits = self.mlp(latent)
        return td.Categorical(logits=logits)


class DreamerV3CriticEnsemble(nn.Module):
    """
    Ensemble of critic networks for improved stability.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_critics: int = 2,
        hidden_dims: list = [512, 512],
        num_bins: int = 255,
        value_low: float = -20.0,
        value_high: float = 20.0,
    ):
        """
        Initialize critic ensemble.
        
        Args:
            latent_dim: Dimension of latent state
            num_critics: Number of critics in ensemble
            hidden_dims: Hidden layer dimensions for each critic
            num_bins: Number of bins for two-hot value encoding
            value_low: Lower bound for value range (in symlog space)
            value_high: Upper bound for value range (in symlog space)
        """
        super().__init__()
        
        self.num_critics = num_critics
        
        self.critics = nn.ModuleList([
            DreamerV3Critic(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                num_bins=num_bins,
                value_low=value_low,
                value_high=value_high,
            )
            for _ in range(num_critics)
        ])
    
    def forward(self, latent: torch.Tensor, reduce: str = 'min') -> torch.Tensor:
        """
        Forward pass through all critics.
        
        Args:
            latent: Latent state
            reduce: How to combine critic predictions ('min', 'mean', 'none')
            
        Returns:
            Value estimate(s)
        """
        values = torch.stack([critic(latent) for critic in self.critics], dim=0)
        
        if reduce == 'min':
            return values.min(dim=0)[0]
        elif reduce == 'mean':
            return values.mean(dim=0)
        elif reduce == 'none':
            return values
        else:
            raise ValueError(f"Unknown reduce mode: {reduce}")
    
    def get_all_values(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Get values from all critics.
        
        Args:
            latent: Latent state
            
        Returns:
            Values from all critics, shape (num_critics, batch)
        """
        return self.forward(latent, reduce='none')


class ActorCriticTrainer:
    """
    Trainer for actor and critic networks via imagination.
    """
    
    def __init__(
        self,
        actor: DreamerV3Actor,
        critic: DreamerV3CriticEnsemble,
        actor_lr: float = 3e-5,
        critic_lr: float = 3e-5,
        actor_grad_clip: float = 100.0,
        critic_grad_clip: float = 100.0,
        discount: float = 0.997,
        lambda_: float = 0.95,
        entropy_coef: float = 3e-4,
        target_update_freq: int = 1,
        target_update_tau: float = 0.02,
    ):
        """
        Initialize actor-critic trainer.
        
        Args:
            actor: Actor network
            critic: Critic ensemble
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            actor_grad_clip: Gradient clipping for actor
            critic_grad_clip: Gradient clipping for critic
            discount: Discount factor
            lambda_: λ for TD(λ) returns
            entropy_coef: Entropy bonus coefficient
            target_update_freq: Frequency of target network updates
            target_update_tau: EMA coefficient for target updates
        """
        self.actor = actor
        self.critic = critic
        self.discount = discount
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef
        self.actor_grad_clip = actor_grad_clip
        self.critic_grad_clip = critic_grad_clip
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr, eps=1e-5)
        
        # Target critic for stable training (copy structure from critic)
        from copy import deepcopy
        self.target_critic = deepcopy(critic)
        self.target_critic.load_state_dict(critic.state_dict())
        # Freeze target critic
        for param in self.target_critic.parameters():
            param.requires_grad = False
        self.target_update_freq = target_update_freq
        self.target_update_tau = target_update_tau
        self.update_step = 0
    
    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute λ-returns for imagination rollout.
        
        Args:
            rewards: Imagined rewards (in symlog space), shape (batch, horizon)
            values: Value estimates (in symlog space), shape (batch, horizon + 1)
            continues: Continue predictions, shape (batch, horizon)
            
        Returns:
            λ-returns (in symlog space), shape (batch, horizon)
        """
        horizon = rewards.shape[1]
        
        # Compute λ-returns using backward recursion
        returns = torch.zeros_like(rewards)
        next_value = values[:, -1]
        
        for t in reversed(range(horizon)):
            # TD target in symlog space
            td_target = rewards[:, t] + continues[:, t] * self.discount * next_value
            
            # λ-return
            returns[:, t] = td_target + continues[:, t] * self.discount * self.lambda_ * \
                           (returns[:, t + 1] if t < horizon - 1 else 0.0)
            next_value = returns[:, t]
        
        return returns
    
    def train_step(
        self,
        latents: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step for actor and critic.
        
        Args:
            latents: Latent states from imagination, shape (batch, horizon, latent_dim)
            rewards: Imagined rewards (in symlog space), shape (batch, horizon)
            continues: Continue predictions, shape (batch, horizon)
            
        Returns:
            Dictionary of training metrics
        """
        batch_size, horizon, latent_dim = latents.shape
        
        # Flatten batch and time dimensions
        latents_flat = latents.reshape(-1, latent_dim)
        
        # Get value estimates
        with torch.no_grad():
            values_all = self.target_critic.get_all_values(latents_flat)
            values_all = values_all.reshape(self.critic.num_critics, batch_size, horizon)
            # Use minimum over ensemble for conservative estimates
            values = values_all.min(dim=0)[0]
        
        # Compute λ-returns
        # Need to append bootstrap value (last value from sequence)
        bootstrap_latent = latents[:, -1]
        with torch.no_grad():
            bootstrap_value = self.target_critic(bootstrap_latent, reduce='min')
        
        values_with_bootstrap = torch.cat([values, bootstrap_value.unsqueeze(1)], dim=1)
        lambda_returns = self.compute_lambda_returns(rewards, values_with_bootstrap, continues)
        
        # Flatten returns
        lambda_returns_flat = lambda_returns.reshape(-1)
        
        # ===== Train Critic =====
        self.critic_optimizer.zero_grad()
        
        # Get current value estimates from all critics
        critic_values = self.critic.get_all_values(latents_flat[:batch_size * horizon])
        critic_values = critic_values.reshape(self.critic.num_critics, -1)
        
        # Critic loss: MSE between values and λ-returns (in symlog space)
        critic_loss = 0.0
        for i in range(self.critic.num_critics):
            critic_loss += F.mse_loss(critic_values[i], lambda_returns_flat)
        critic_loss = critic_loss / self.critic.num_critics
        
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.critic_grad_clip
        )
        self.critic_optimizer.step()
        
        # ===== Train Actor =====
        self.actor_optimizer.zero_grad()
        
        # Get actions and log probs
        actions, log_probs = self.actor(latents_flat[:batch_size * horizon])
        
        # Compute advantages (returns - baseline)
        with torch.no_grad():
            baseline = self.critic(latents_flat[:batch_size * horizon], reduce='mean')
        advantages = lambda_returns_flat - baseline
        
        # Actor loss: -E[advantages * log_prob] - entropy_bonus
        _, entropy = self.actor.evaluate_actions(latents_flat[:batch_size * horizon], actions)
        actor_loss = -(advantages.detach() * log_probs).mean() - self.entropy_coef * entropy.mean()
        
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.actor_grad_clip
        )
        self.actor_optimizer.step()
        
        # Update target network
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.update_target()
        
        # Return metrics
        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'entropy': entropy.mean().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'returns_mean': lambda_returns_flat.mean().item(),
            'returns_std': lambda_returns_flat.std().item(),
        }
        
        return metrics
    
    def update_target(self):
        """Update target critic with EMA."""
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    self.target_update_tau * param.data + 
                    (1.0 - self.target_update_tau) * target_param.data
                )

