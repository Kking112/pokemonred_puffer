"""
Tests for DreamerV3 actor and critic networks.
"""

import pytest
import torch

from pokemonred_puffer.models.dreamer.actor_critic import (
    DreamerV3Actor,
    DreamerV3Critic,
    DreamerV3CriticEnsemble,
    ActorCriticTrainer,
)


class TestDreamerV3Actor:
    """Test actor network."""
    
    @pytest.fixture
    def actor(self):
        """Create actor for testing."""
        return DreamerV3Actor(
            latent_dim=512,
            action_dim=7,
            hidden_dims=[256, 256],
            discrete=True,
        )
    
    def test_actor_forward_discrete(self, actor):
        """Test actor forward pass for discrete actions."""
        batch_size = 4
        latent = torch.randn(batch_size, 512)
        
        actions, log_probs = actor(latent, deterministic=False)
        
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < 7)
    
    def test_actor_deterministic(self, actor):
        """Test deterministic action selection."""
        latent = torch.randn(1, 512)
        
        # Get deterministic action multiple times
        action1, _ = actor(latent, deterministic=True)
        action2, _ = actor(latent, deterministic=True)
        
        # Should be the same (mode)
        assert action1 == action2
    
    def test_actor_evaluate_actions(self, actor):
        """Test evaluating given actions."""
        batch_size = 4
        latent = torch.randn(batch_size, 512)
        actions = torch.randint(0, 7, (batch_size,))
        
        log_probs, entropy = actor.evaluate_actions(latent, actions)
        
        assert log_probs.shape == (batch_size,)
        assert entropy.shape == (batch_size,)


class TestDreamerV3Critic:
    """Test critic network."""
    
    @pytest.fixture
    def critic(self):
        """Create critic for testing."""
        return DreamerV3Critic(
            latent_dim=512,
            hidden_dims=[256, 256],
        )
    
    def test_critic_forward(self, critic):
        """Test critic forward pass."""
        batch_size = 4
        latent = torch.randn(batch_size, 512)
        
        value = critic(latent)
        
        assert value.shape == (batch_size,)
    
    def test_critic_get_value_dist(self, critic):
        """Test getting value distribution."""
        latent = torch.randn(4, 512)
        
        dist = critic.get_value_dist(latent)
        
        # Should return a categorical distribution
        assert hasattr(dist, 'sample')
        assert hasattr(dist, 'log_prob')


class TestDreamerV3CriticEnsemble:
    """Test critic ensemble."""
    
    @pytest.fixture
    def critic_ensemble(self):
        """Create critic ensemble for testing."""
        return DreamerV3CriticEnsemble(
            latent_dim=512,
            num_critics=3,
            hidden_dims=[256, 256],
        )
    
    def test_ensemble_forward_min(self, critic_ensemble):
        """Test ensemble forward with min reduction."""
        latent = torch.randn(4, 512)
        
        value = critic_ensemble(latent, reduce='min')
        
        assert value.shape == (4,)
    
    def test_ensemble_forward_mean(self, critic_ensemble):
        """Test ensemble forward with mean reduction."""
        latent = torch.randn(4, 512)
        
        value = critic_ensemble(latent, reduce='mean')
        
        assert value.shape == (4,)
    
    def test_ensemble_forward_none(self, critic_ensemble):
        """Test ensemble forward without reduction."""
        latent = torch.randn(4, 512)
        
        values = critic_ensemble(latent, reduce='none')
        
        assert values.shape == (3, 4)  # (num_critics, batch)
    
    def test_get_all_values(self, critic_ensemble):
        """Test getting values from all critics."""
        latent = torch.randn(4, 512)
        
        values = critic_ensemble.get_all_values(latent)
        
        assert values.shape == (3, 4)


class TestActorCriticTrainer:
    """Test actor-critic trainer."""
    
    @pytest.fixture
    def trainer(self):
        """Create trainer for testing."""
        actor = DreamerV3Actor(
            latent_dim=256,
            action_dim=7,
            hidden_dims=[128],
            discrete=True,
        )
        
        critic = DreamerV3CriticEnsemble(
            latent_dim=256,
            num_critics=2,
            hidden_dims=[128],
        )
        
        return ActorCriticTrainer(
            actor=actor,
            critic=critic,
            actor_lr=3e-5,
            critic_lr=3e-5,
        )
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.actor is not None
        assert trainer.critic is not None
        assert trainer.actor_optimizer is not None
        assert trainer.critic_optimizer is not None
        assert trainer.target_critic is not None
    
    def test_compute_lambda_returns(self, trainer):
        """Test computing lambda returns."""
        batch_size = 4
        horizon = 10
        
        rewards = torch.randn(batch_size, horizon)
        values = torch.randn(batch_size, horizon + 1)
        continues = torch.ones(batch_size, horizon)
        
        returns = trainer.compute_lambda_returns(rewards, values, continues)
        
        assert returns.shape == (batch_size, horizon)
    
    def test_train_step(self, trainer):
        """Test single training step."""
        batch_size = 4
        horizon = 10
        latent_dim = 256
        
        latents = torch.randn(batch_size, horizon, latent_dim)
        rewards = torch.randn(batch_size, horizon)
        continues = torch.ones(batch_size, horizon)
        
        metrics = trainer.train_step(latents, rewards, continues)
        
        # Check metrics
        assert 'actor_loss' in metrics
        assert 'critic_loss' in metrics
        assert 'entropy' in metrics
        assert 'advantages_mean' in metrics
        assert 'returns_mean' in metrics
    
    def test_update_target(self, trainer):
        """Test target network update."""
        # Get initial target params
        initial_params = [p.clone() for p in trainer.target_critic.parameters()]
        
        # Modify the critic parameters (simulate training)
        for p in trainer.critic.parameters():
            p.data += 0.1 * torch.randn_like(p)
        
        # Update target
        trainer.update_target()
        
        # Check that parameters changed (but only slightly due to EMA)
        updated_params = list(trainer.target_critic.parameters())
        for init_p, updated_p in zip(initial_params, updated_params):
            # Should be different (EMA update)
            # But not drastically different
            assert not torch.equal(init_p, updated_p)
            # Check that the update is small (EMA with tau=0.01)
            diff = torch.abs(updated_p - init_p).max()
            assert diff < 0.01  # Should be small change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

