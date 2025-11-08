"""
Replay Buffer for DreamerV3.

Stores sequences of experiences for training the world model and actor-critic.
Supports efficient storage and sampling with multi-environment parallelism.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from collections import defaultdict


class SequenceReplayBuffer:
    """
    Replay buffer that stores and samples sequences of experiences.
    
    Optimized for DreamerV3's needs:
    - Stores full observation dictionaries
    - Supports sequence sampling
    - Efficient storage with uint8 for images
    - Handles multiple parallel environments
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shapes: Dict[str, Tuple],
        observation_dtypes: Dict[str, np.dtype],
        action_dim: int,
        sequence_length: int = 50,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of timesteps to store
            observation_shapes: Dictionary mapping observation names to shapes
            observation_dtypes: Dictionary mapping observation names to dtypes
            action_dim: Dimension of action space
            sequence_length: Length of sequences to sample
            batch_size: Number of sequences to sample per batch
            num_workers: Number of parallel workers for data loading
        """
        self.capacity = capacity
        self.observation_shapes = observation_shapes
        self.observation_dtypes = observation_dtypes
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Storage
        self.observations = {}
        for name, shape in observation_shapes.items():
            dtype = observation_dtypes[name]
            self.observations[name] = np.zeros(
                (capacity, *shape), dtype=dtype
            )
        
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        
        # Pointer and size
        self.ptr = 0
        self.size = 0
        
        # Episode boundaries (for avoiding sampling across episodes)
        self.episode_starts = []
        self.episode_ends = []
        self.current_episode_start = 0
    
    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """
        Add a single timestep to the buffer.
        
        Args:
            observation: Dictionary of observations
            action: Action taken
            reward: Reward received
            done: Whether episode ended
        """
        # Store observation
        for name, obs in observation.items():
            if name in self.observations:
                self.observations[name][self.ptr] = obs
        
        # Store action, reward, done
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        # Track episode boundaries
        if done:
            self.episode_ends.append(self.ptr)
            if len(self.episode_starts) <= len(self.episode_ends):
                # Record start of next episode
                next_start = (self.ptr + 1) % self.capacity
                self.episode_starts.append(next_start)
                self.current_episode_start = next_start
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Clean up old episode boundaries
        if self.size == self.capacity:
            # Remove episode boundaries that fall outside the buffer
            self.episode_starts = [s for s in self.episode_starts if s < self.capacity]
            self.episode_ends = [e for e in self.episode_ends if e < self.capacity]
    
    def add_batch(
        self,
        observations: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        """
        Add a batch of timesteps to the buffer.
        
        Args:
            observations: Dictionary of observation arrays, each (batch, ...)
            actions: Actions, shape (batch, action_dim)
            rewards: Rewards, shape (batch,)
            dones: Dones, shape (batch,)
        """
        batch_size = actions.shape[0]
        
        for i in range(batch_size):
            obs_i = {name: obs[i] for name, obs in observations.items()}
            self.add(obs_i, actions[i], rewards[i], dones[i])
    
    def sample_sequences(self, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        Sample a batch of sequences from the buffer.
        
        Args:
            device: Device to load tensors to
            
        Returns:
            Dictionary containing:
            - observations: Dict of observation sequences, each (batch, seq_len, ...)
            - actions: Action sequences, (batch, seq_len, action_dim)
            - rewards: Reward sequences, (batch, seq_len)
            - dones: Done flags, (batch, seq_len)
        """
        if self.size < self.sequence_length:
            raise ValueError(f"Buffer has {self.size} timesteps, need at least {self.sequence_length}")
        
        # Sample random starting indices
        # Ensure we don't sample across episode boundaries
        valid_starts = []
        for _ in range(self.batch_size * 10):  # Over-sample and filter
            start_idx = np.random.randint(0, self.size - self.sequence_length + 1)
            end_idx = start_idx + self.sequence_length
            
            # Check if sequence crosses episode boundary
            crosses_boundary = False
            for ep_end in self.episode_ends:
                if start_idx <= ep_end < end_idx:
                    crosses_boundary = True
                    break
            
            if not crosses_boundary:
                valid_starts.append(start_idx)
                if len(valid_starts) >= self.batch_size:
                    break
        
        if len(valid_starts) < self.batch_size:
            # Fall back to sampling without boundary check if needed
            valid_starts = np.random.randint(0, self.size - self.sequence_length + 1, self.batch_size)
        else:
            valid_starts = valid_starts[:self.batch_size]
        
        # Extract sequences
        observations = {}
        for name, obs_buffer in self.observations.items():
            sequences = []
            for start in valid_starts:
                seq = obs_buffer[start:start + self.sequence_length]
                sequences.append(seq)
            
            obs_array = np.stack(sequences, axis=0)
            observations[name] = torch.from_numpy(obs_array).to(device)
        
        # Extract actions, rewards, dones
        action_sequences = []
        reward_sequences = []
        done_sequences = []
        
        for start in valid_starts:
            action_sequences.append(self.actions[start:start + self.sequence_length])
            reward_sequences.append(self.rewards[start:start + self.sequence_length])
            done_sequences.append(self.dones[start:start + self.sequence_length])
        
        actions = torch.from_numpy(np.stack(action_sequences, axis=0)).to(device)
        rewards = torch.from_numpy(np.stack(reward_sequences, axis=0)).to(device)
        dones = torch.from_numpy(np.stack(done_sequences, axis=0)).to(device)
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
        }
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data to sample sequences."""
        return self.size >= self.sequence_length * self.batch_size


class MultiEnvReplayBuffer:
    """
    Replay buffer that handles multiple parallel environments.
    
    This maintains separate episode tracking for each environment and
    aggregates data from all environments into a shared buffer.
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shapes: Dict[str, Tuple],
        observation_dtypes: Dict[str, np.dtype],
        action_dim: int,
        num_envs: int,
        sequence_length: int = 50,
        batch_size: int = 16,
    ):
        """
        Initialize multi-environment replay buffer.
        
        Args:
            capacity: Total capacity across all environments
            observation_shapes: Dictionary mapping observation names to shapes
            observation_dtypes: Dictionary mapping observation names to dtypes
            action_dim: Dimension of action space
            num_envs: Number of parallel environments
            sequence_length: Length of sequences to sample
            batch_size: Number of sequences per batch
        """
        self.num_envs = num_envs
        self.capacity_per_env = capacity // num_envs
        
        # Create a separate buffer for each environment
        self.buffers = [
            SequenceReplayBuffer(
                capacity=self.capacity_per_env,
                observation_shapes=observation_shapes,
                observation_dtypes=observation_dtypes,
                action_dim=action_dim,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
            for _ in range(num_envs)
        ]
        
        self.sequence_length = sequence_length
        self.batch_size = batch_size
    
    def add(
        self,
        env_id: int,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """
        Add experience from a specific environment.
        
        Args:
            env_id: Environment ID
            observation: Observation dictionary
            action: Action taken
            reward: Reward received
            done: Whether episode ended
        """
        self.buffers[env_id].add(observation, action, reward, done)
    
    def add_batch(
        self,
        observations: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        env_ids: Optional[np.ndarray] = None,
    ):
        """
        Add a batch of experiences from multiple environments.
        
        Args:
            observations: Dictionary of observation arrays
            actions: Actions
            rewards: Rewards
            dones: Dones
            env_ids: Environment IDs (if None, assumes sequential ordering)
        """
        batch_size = actions.shape[0]
        
        if env_ids is None:
            env_ids = np.arange(batch_size) % self.num_envs
        
        for i in range(batch_size):
            env_id = env_ids[i]
            obs_i = {name: obs[i] for name, obs in observations.items()}
            self.buffers[env_id].add(obs_i, actions[i], rewards[i], dones[i])
    
    def sample_sequences(self, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        Sample sequences from all environment buffers.
        
        The batch size is distributed across all environments.
        
        Args:
            device: Device to load tensors to
            
        Returns:
            Dictionary of sampled sequences
        """
        # Determine how many sequences to sample from each environment
        sequences_per_env = self.batch_size // self.num_envs
        remainder = self.batch_size % self.num_envs
        
        # Sample from each environment
        all_observations = defaultdict(list)
        all_actions = []
        all_rewards = []
        all_dones = []
        
        for i, buffer in enumerate(self.buffers):
            if not buffer.is_ready():
                continue
            
            # Adjust batch size for this buffer
            buffer_batch_size = sequences_per_env + (1 if i < remainder else 0)
            
            # Temporarily set batch size
            original_batch_size = buffer.batch_size
            buffer.batch_size = buffer_batch_size
            
            try:
                batch = buffer.sample_sequences(device)
                
                for name, obs in batch['observations'].items():
                    all_observations[name].append(obs)
                
                all_actions.append(batch['actions'])
                all_rewards.append(batch['rewards'])
                all_dones.append(batch['dones'])
            finally:
                # Restore original batch size
                buffer.batch_size = original_batch_size
        
        if not all_actions:
            raise ValueError("No environments have enough data to sample")
        
        # Concatenate all samples
        observations = {
            name: torch.cat(obs_list, dim=0) 
            for name, obs_list in all_observations.items()
        }
        actions = torch.cat(all_actions, dim=0)
        rewards = torch.cat(all_rewards, dim=0)
        dones = torch.cat(all_dones, dim=0)
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
        }
    
    def __len__(self) -> int:
        """Return total size across all buffers."""
        return sum(len(buffer) for buffer in self.buffers)
    
    def is_ready(self) -> bool:
        """Check if at least one buffer is ready to sample."""
        return any(buffer.is_ready() for buffer in self.buffers)
    
    def total_capacity(self) -> int:
        """Return total capacity across all buffers."""
        return self.capacity_per_env * self.num_envs

