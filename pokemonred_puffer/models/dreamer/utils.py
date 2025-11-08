"""
Utility functions for DreamerV3 implementation.

Includes symlog transforms, distribution utilities, and other helper functions
specific to DreamerV3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as td
from typing import Dict, Any, Optional


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric log transform used in DreamerV3.
    
    symlog(x) = sign(x) * log(|x| + 1)
    
    This transform handles both positive and negative values and has
    a smooth gradient near zero.
    
    Args:
        x: Input tensor
        
    Returns:
        Transformed tensor
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of symlog transform.
    
    symexp(x) = sign(x) * (exp(|x|) - 1)
    
    Args:
        x: Input tensor (in symlog space)
        
    Returns:
        Transformed tensor (in original space)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def two_hot_encode(x: torch.Tensor, num_bins: int = 255, 
                    low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """
    Two-hot encoding for continuous values.
    
    This encoding represents a continuous value as a soft distribution over
    discrete bins, with linear interpolation between adjacent bins.
    Used in DreamerV3 for representing discrete actions and predictions.
    
    Args:
        x: Input values to encode, shape (...)
        num_bins: Number of bins
        low: Lower bound of value range
        high: Upper bound of value range
        
    Returns:
        Two-hot encoded tensor, shape (..., num_bins)
    """
    # Clip to range
    x = torch.clamp(x, low, high)
    
    # Normalize to [0, num_bins - 1]
    x_norm = (x - low) / (high - low) * (num_bins - 1)
    
    # Get lower and upper bin indices
    x_low = torch.floor(x_norm).long()
    x_high = torch.ceil(x_norm).long()
    
    # Clamp indices
    x_low = torch.clamp(x_low, 0, num_bins - 1)
    x_high = torch.clamp(x_high, 0, num_bins - 1)
    
    # Compute weights for interpolation
    weight_high = x_norm - x_low.float()
    weight_low = 1.0 - weight_high
    
    # Create one-hot encodings
    shape = x.shape
    one_hot_low = F.one_hot(x_low, num_bins).float()
    one_hot_high = F.one_hot(x_high, num_bins).float()
    
    # Weight and combine
    weight_low = weight_low.unsqueeze(-1)
    weight_high = weight_high.unsqueeze(-1)
    
    two_hot = weight_low * one_hot_low + weight_high * one_hot_high
    
    return two_hot


def two_hot_decode(probs: torch.Tensor, num_bins: int = 255,
                    low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """
    Decode two-hot encoding to continuous value.
    
    Args:
        probs: Probability distribution over bins, shape (..., num_bins)
        num_bins: Number of bins
        low: Lower bound of value range
        high: Upper bound of value range
        
    Returns:
        Decoded continuous values, shape (...)
    """
    # Create bin centers
    bins = torch.linspace(low, high, num_bins, device=probs.device)
    
    # Expected value
    value = (probs * bins).sum(dim=-1)
    
    return value


class CategoricalDistribution:
    """
    Helper class for categorical distributions used in DreamerV3's stochastic states.
    
    This wraps PyTorch's Categorical distribution with additional utilities.
    """
    
    def __init__(self, logits: Optional[torch.Tensor] = None,
                 probs: Optional[torch.Tensor] = None,
                 unimix: float = 0.01):
        """
        Initialize categorical distribution.
        
        Args:
            logits: Logits for categorical distribution
            probs: Probabilities for categorical distribution
            unimix: Uniform mixture weight for preventing mode collapse
        """
        if logits is not None:
            if unimix > 0.0:
                # Mix with uniform distribution
                uniform_logits = torch.ones_like(logits) / logits.shape[-1]
                logits = (1.0 - unimix) * F.softmax(logits, dim=-1) + unimix * uniform_logits
                logits = torch.log(logits + 1e-8)
            self.dist = td.Categorical(logits=logits)
        elif probs is not None:
            if unimix > 0.0:
                uniform_probs = torch.ones_like(probs) / probs.shape[-1]
                probs = (1.0 - unimix) * probs + unimix * uniform_probs
            self.dist = td.Categorical(probs=probs)
        else:
            raise ValueError("Either logits or probs must be provided")
    
    def sample(self) -> torch.Tensor:
        """Sample from distribution."""
        return self.dist.sample()
    
    def mode(self) -> torch.Tensor:
        """Get mode (most likely value)."""
        return self.dist.probs.argmax(dim=-1)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of value."""
        return self.dist.log_prob(value)
    
    def entropy(self) -> torch.Tensor:
        """Compute entropy."""
        return self.dist.entropy()
    
    @property
    def probs(self) -> torch.Tensor:
        """Get probabilities."""
        return self.dist.probs


class DenseLayer(nn.Module):
    """
    Dense layer with activation and normalization.
    
    This is a building block used throughout DreamerV3.
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 activation: str = 'silu', norm: bool = True):
        """
        Initialize dense layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function ('silu', 'relu', 'gelu', 'elu', or 'none')
            norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        
        if norm:
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = nn.Identity()
        
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron used in DreamerV3.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list[int], 
                 output_dim: int, activation: str = 'silu', 
                 output_activation: str = 'none', norm: bool = True):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            norm: Whether to use layer normalization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(DenseLayer(prev_dim, hidden_dim, activation, norm))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(DenseLayer(prev_dim, output_dim, output_activation, norm=False))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """
    Compute λ-returns for value learning.
    
    This implements the TD(λ) return computation used in DreamerV3.
    
    Args:
        rewards: Rewards, shape (batch, time)
        values: Value predictions, shape (batch, time)
        continues: Continue flags (1 - done), shape (batch, time)
        bootstrap: Bootstrap value for the last timestep, shape (batch,)
        lambda_: λ parameter for mixing TD targets
        
    Returns:
        λ-returns, shape (batch, time)
    """
    # Append bootstrap value
    values_with_bootstrap = torch.cat([values, bootstrap.unsqueeze(1)], dim=1)
    
    # Compute TD errors
    deltas = rewards + continues * values_with_bootstrap[:, 1:] - values_with_bootstrap[:, :-1]
    
    # Compute λ-returns via backward recursion
    returns = torch.zeros_like(values)
    last = bootstrap
    
    for t in reversed(range(rewards.shape[1])):
        returns[:, t] = deltas[:, t] + continues[:, t] * lambda_ * last
        last = returns[:, t]
    
    return returns


def static_scan(fn, inputs, start):
    """
    Static scan operation for unrolling sequences.
    
    This is used for imagination rollouts in DreamerV3.
    
    Args:
        fn: Function to apply at each step
        inputs: Sequence of inputs
        start: Initial state
        
    Returns:
        Sequence of outputs and final state
    """
    last = start
    outputs = []
    
    for input_t in inputs:
        last = fn(last, input_t)
        outputs.append(last)
    
    return outputs, last

