"""
Tests for replay buffer.
"""

import pytest
import numpy as np
import torch

from pokemonred_puffer.models.dreamer.replay_buffer import (
    SequenceReplayBuffer,
    MultiEnvReplayBuffer,
)


class TestSequenceReplayBuffer:
    """Test sequence replay buffer."""
    
    @pytest.fixture
    def buffer(self):
        """Create replay buffer for testing."""
        observation_shapes = {
            'screen': (72, 80, 1),
            'text': (200,),
            'direction': (1,),
        }
        observation_dtypes = {
            'screen': np.uint8,
            'text': np.uint8,
            'direction': np.uint8,
        }
        
        return SequenceReplayBuffer(
            capacity=1000,
            observation_shapes=observation_shapes,
            observation_dtypes=observation_dtypes,
            action_dim=7,
            sequence_length=10,
            batch_size=4,
        )
    
    def test_buffer_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.capacity == 1000
        assert buffer.sequence_length == 10
        assert buffer.batch_size == 4
        assert buffer.size == 0
        assert buffer.ptr == 0
    
    def test_add_single_experience(self, buffer):
        """Test adding single experience."""
        obs = {
            'screen': np.random.randint(0, 256, (72, 80, 1), dtype=np.uint8),
            'text': np.random.randint(0, 256, (200,), dtype=np.uint8),
            'direction': np.array([0], dtype=np.uint8),
        }
        action = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        reward = 1.0
        done = False
        
        buffer.add(obs, action, reward, done)
        
        assert buffer.size == 1
        assert buffer.ptr == 1
    
    def test_add_multiple_experiences(self, buffer):
        """Test adding multiple experiences."""
        for i in range(50):
            obs = {
                'screen': np.random.randint(0, 256, (72, 80, 1), dtype=np.uint8),
                'text': np.random.randint(0, 256, (200,), dtype=np.uint8),
                'direction': np.array([i % 4], dtype=np.uint8),
            }
            action = np.random.randn(7).astype(np.float32)
            reward = float(i)
            done = (i % 10 == 9)
            
            buffer.add(obs, action, reward, done)
        
        assert buffer.size == 50
        assert len(buffer) == 50
    
    def test_buffer_wraps_around(self, buffer):
        """Test that buffer wraps around at capacity."""
        # Fill buffer beyond capacity
        for i in range(1100):
            obs = {
                'screen': np.random.randint(0, 256, (72, 80, 1), dtype=np.uint8),
                'text': np.random.randint(0, 256, (200,), dtype=np.uint8),
                'direction': np.array([0], dtype=np.uint8),
            }
            action = np.random.randn(7).astype(np.float32)
            buffer.add(obs, action, 0.0, False)
        
        assert buffer.size == 1000  # Should be capped at capacity
        assert buffer.ptr == 100  # Should have wrapped around
    
    def test_sample_sequences(self, buffer):
        """Test sampling sequences."""
        # Fill buffer with data
        for i in range(100):
            obs = {
                'screen': np.random.randint(0, 256, (72, 80, 1), dtype=np.uint8),
                'text': np.random.randint(0, 256, (200,), dtype=np.uint8),
                'direction': np.array([0], dtype=np.uint8),
            }
            action = np.random.randn(7).astype(np.float32)
            buffer.add(obs, action, 0.0, False)
        
        # Sample batch
        batch = buffer.sample_sequences(device='cpu')
        
        assert 'observations' in batch
        assert 'actions' in batch
        assert 'rewards' in batch
        assert 'dones' in batch
        
        # Check shapes
        assert batch['actions'].shape == (4, 10, 7)  # (batch, seq, action_dim)
        assert batch['rewards'].shape == (4, 10)
        assert batch['dones'].shape == (4, 10)
        assert batch['observations']['screen'].shape == (4, 10, 72, 80, 1)
    
    def test_is_ready(self, buffer):
        """Test is_ready method."""
        assert not buffer.is_ready()
        
        # Add enough data
        for i in range(50):
            obs = {
                'screen': np.random.randint(0, 256, (72, 80, 1), dtype=np.uint8),
                'text': np.random.randint(0, 256, (200,), dtype=np.uint8),
                'direction': np.array([0], dtype=np.uint8),
            }
            action = np.random.randn(7).astype(np.float32)
            buffer.add(obs, action, 0.0, False)
        
        assert buffer.is_ready()


class TestMultiEnvReplayBuffer:
    """Test multi-environment replay buffer."""
    
    @pytest.fixture
    def multi_buffer(self):
        """Create multi-env replay buffer for testing."""
        observation_shapes = {
            'screen': (72, 80, 1),
            'text': (200,),
        }
        observation_dtypes = {
            'screen': np.uint8,
            'text': np.uint8,
        }
        
        return MultiEnvReplayBuffer(
            capacity=1000,
            observation_shapes=observation_shapes,
            observation_dtypes=observation_dtypes,
            action_dim=7,
            num_envs=4,
            sequence_length=10,
            batch_size=8,
        )
    
    def test_multi_buffer_initialization(self, multi_buffer):
        """Test multi-env buffer initialization."""
        assert multi_buffer.num_envs == 4
        assert len(multi_buffer.buffers) == 4
        assert multi_buffer.total_capacity() == 1000
    
    def test_add_to_specific_env(self, multi_buffer):
        """Test adding experience to specific environment."""
        obs = {
            'screen': np.random.randint(0, 256, (72, 80, 1), dtype=np.uint8),
            'text': np.random.randint(0, 256, (200,), dtype=np.uint8),
        }
        action = np.random.randn(7).astype(np.float32)
        
        multi_buffer.add(env_id=2, observation=obs, action=action, reward=1.0, done=False)
        
        assert multi_buffer.buffers[2].size == 1
        assert multi_buffer.buffers[0].size == 0
    
    def test_add_batch(self, multi_buffer):
        """Test adding batch of experiences."""
        batch_size = 8
        
        observations = {
            'screen': np.random.randint(0, 256, (batch_size, 72, 80, 1), dtype=np.uint8),
            'text': np.random.randint(0, 256, (batch_size, 200), dtype=np.uint8),
        }
        actions = np.random.randn(batch_size, 7).astype(np.float32)
        rewards = np.random.randn(batch_size).astype(np.float32)
        dones = np.zeros(batch_size, dtype=bool)
        env_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        
        multi_buffer.add_batch(observations, actions, rewards, dones, env_ids)
        
        # Each env should have 2 experiences
        for i in range(4):
            assert multi_buffer.buffers[i].size == 2
    
    def test_sample_from_multiple_envs(self, multi_buffer):
        """Test sampling from multiple environments."""
        # Fill buffers - need sequence_length * batch_size data points
        # multi_buffer has sequence_length=10, batch_size=8
        # So each buffer needs 10 * 8 = 80 data points
        for env_id in range(4):
            for i in range(85):  # Add enough data: 85 > 80
                obs = {
                    'screen': np.random.randint(0, 256, (72, 80, 1), dtype=np.uint8),
                    'text': np.random.randint(0, 256, (200,), dtype=np.uint8),
                }
                action = np.random.randn(7).astype(np.float32)
                multi_buffer.add(env_id, obs, action, 0.0, False)
        
        # Check each buffer is ready
        assert all(buf.is_ready() for buf in multi_buffer.buffers)
        
        # Sample
        batch = multi_buffer.sample_sequences(device='cpu')
        
        # Should get batch_size sequences
        assert batch['actions'].shape[0] == 8
        assert batch['actions'].shape[1] == 10  # sequence_length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

