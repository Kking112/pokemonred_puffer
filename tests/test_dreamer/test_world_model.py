"""
Tests for DreamerV3 world model components.
"""

import pytest
import torch
import numpy as np

from pokemonred_puffer.models.dreamer.world_model import (
    RSSM,
    ObservationEncoder,
    ObservationDecoder,
    RewardPredictor,
    ContinuePredictor,
    DreamerV3WorldModel,
)
from pokemonred_puffer.models.text_encoder import TransformerTextEncoder


class TestRSSM:
    """Test RSSM (Recurrent State-Space Model)."""
    
    @pytest.fixture
    def rssm(self):
        """Create RSSM instance for testing."""
        return RSSM(
            action_dim=7,
            deter_dim=256,
            stoch_dim=16,
            stoch_classes=16,
            hidden_dim=256,
        )
    
    def test_rssm_initialization(self, rssm):
        """Test RSSM initialization."""
        assert rssm.action_dim == 7
        assert rssm.deter_dim == 256
        assert rssm.stoch_dim == 16
        assert rssm.stoch_classes == 16
        assert rssm.stoch_total == 16 * 16
    
    def test_initial_state(self, rssm):
        """Test creating initial state."""
        batch_size = 4
        device = torch.device('cpu')
        
        state = rssm.initial_state(batch_size, device)
        
        assert 'deter' in state
        assert 'stoch' in state
        assert state['deter'].shape == (batch_size, 256)
        assert state['stoch'].shape == (batch_size, 16, 16)
    
    def test_imagine_step(self, rssm):
        """Test single imagination step."""
        batch_size = 4
        device = torch.device('cpu')
        
        state = rssm.initial_state(batch_size, device)
        action = torch.randn(batch_size, 7)
        
        new_state, prior_dist = rssm.imagine_step(state, action)
        
        assert 'deter' in new_state
        assert 'stoch' in new_state
        assert new_state['deter'].shape == (batch_size, 256)
        assert new_state['stoch'].shape == (batch_size, 16, 16)
    
    def test_observe_step(self, rssm):
        """Test observation step with posterior."""
        batch_size = 4
        device = torch.device('cpu')
        
        state = rssm.initial_state(batch_size, device)
        action = torch.randn(batch_size, 7)
        obs_embedding = torch.randn(batch_size, 256)
        
        new_state, prior_dist, posterior_dist = rssm.observe_step(
            state, action, obs_embedding
        )
        
        assert 'deter' in new_state
        assert 'stoch' in new_state
        assert new_state['deter'].shape == (batch_size, 256)
        assert new_state['stoch'].shape == (batch_size, 16, 16)
    
    def test_observe_sequence(self, rssm):
        """Test processing a sequence of observations."""
        batch_size = 4
        seq_len = 10
        device = torch.device('cpu')
        
        obs_embeddings = torch.randn(batch_size, seq_len, 256)
        actions = torch.randn(batch_size, seq_len, 7)
        
        states, prior_dists, posterior_dists = rssm.observe_sequence(
            obs_embeddings, actions
        )
        
        assert states['deter'].shape == (batch_size, seq_len, 256)
        assert states['stoch'].shape == (batch_size, seq_len, 16, 16)
        assert len(prior_dists) == seq_len
        assert len(posterior_dists) == seq_len
    
    def test_imagine_sequence(self, rssm):
        """Test imagination rollout."""
        batch_size = 4
        seq_len = 10
        device = torch.device('cpu')
        
        initial_state = rssm.initial_state(batch_size, device)
        actions = torch.randn(batch_size, seq_len, 7)
        
        states, prior_dists = rssm.imagine_sequence(initial_state, actions)
        
        assert states['deter'].shape == (batch_size, seq_len, 256)
        assert states['stoch'].shape == (batch_size, seq_len, 16, 16)
        assert len(prior_dists) == seq_len


class TestRewardAndContinuePredictors:
    """Test reward and continue predictors."""
    
    def test_reward_predictor(self):
        """Test reward predictor."""
        latent_dim = 512
        batch_size = 4
        
        predictor = RewardPredictor(latent_dim)
        latent = torch.randn(batch_size, latent_dim)
        
        reward_pred = predictor(latent)
        
        assert reward_pred.shape == (batch_size,)
    
    def test_continue_predictor(self):
        """Test continue predictor."""
        latent_dim = 512
        batch_size = 4
        
        predictor = ContinuePredictor(latent_dim)
        latent = torch.randn(batch_size, latent_dim)
        
        continue_pred = predictor(latent)
        
        assert continue_pred.shape == (batch_size,)
        assert torch.all(continue_pred >= 0.0)
        assert torch.all(continue_pred <= 1.0)


class TestObservationDecoder:
    """Test observation decoder."""
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        latent_dim = 512
        screen_shape = (72, 80, 1)
        batch_size = 4
        
        decoder = ObservationDecoder(latent_dim, screen_shape)
        latent = torch.randn(batch_size, latent_dim)
        
        screen_recon = decoder(latent)
        
        assert screen_recon.shape == (batch_size, screen_shape[2], screen_shape[0], screen_shape[1])


class TestWorldModelIntegration:
    """Integration tests for complete world model."""
    
    @pytest.fixture
    def world_model(self):
        """Create a small world model for testing."""
        text_encoder = TransformerTextEncoder(
            vocab_size=256,
            embed_dim=64,
            num_heads=2,
            num_layers=1,
            output_dim=64,
        )
        
        # Calculate state_dim: direction(1) + map_id(1) + battle_type(1) + level(6) + hp(6) + events(320) = 335
        return DreamerV3WorldModel(
            action_dim=7,
            screen_shape=(72, 80, 1),
            text_encoder=text_encoder,
            state_dim=335,
            deter_dim=128,
            stoch_dim=8,
            stoch_classes=8,
            hidden_dim=128,
            encoder_dim=128,
        )
    
    def test_world_model_forward(self, world_model):
        """Test world model forward pass."""
        batch_size = 2
        seq_len = 5
        
        # Create dummy observations - use float32 to avoid uint8 normalization issues
        observations = {
            'screen': torch.randint(0, 256, (batch_size, seq_len, 72, 80, 1), dtype=torch.float32),
            'visited_mask': torch.randint(0, 256, (batch_size, seq_len, 72, 80, 1), dtype=torch.float32),
            'text': torch.randint(0, 256, (batch_size, seq_len, 200), dtype=torch.long),  # text should be long for indexing
            'direction': torch.randint(0, 4, (batch_size, seq_len, 1), dtype=torch.float32),
            'map_id': torch.randint(0, 247, (batch_size, seq_len, 1), dtype=torch.float32),
            'battle_type': torch.randint(0, 4, (batch_size, seq_len, 1), dtype=torch.float32),
            'level': torch.randint(1, 100, (batch_size, seq_len, 6), dtype=torch.float32),
            'hp': torch.randint(0, 714, (batch_size, seq_len, 6), dtype=torch.float32),
            'events': torch.randint(0, 2, (batch_size, seq_len, 320), dtype=torch.float32),
        }
        
        # Actions need to be one-hot encoded for world model
        actions_discrete = torch.randint(0, 7, (batch_size, seq_len), dtype=torch.long)
        actions = torch.nn.functional.one_hot(actions_discrete, num_classes=7).float()
        
        # Forward pass
        outputs = world_model(observations, actions)
        
        # Check outputs
        assert 'states' in outputs
        assert 'latents' in outputs
        assert 'prior_dists' in outputs
        assert 'posterior_dists' in outputs
        assert 'screen_recon' in outputs
        assert 'reward_pred' in outputs
        assert 'continue_pred' in outputs
        
        # Check shapes
        assert outputs['latents'].shape[0] == batch_size
        assert outputs['latents'].shape[1] == seq_len
        assert outputs['reward_pred'].shape == (batch_size, seq_len)
        assert outputs['continue_pred'].shape == (batch_size, seq_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

