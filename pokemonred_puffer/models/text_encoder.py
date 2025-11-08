"""
Transformer-based text encoder for Pokemon Red in-game text.

This module provides a transformer encoder to process and understand Pokemon Red's
in-game text, converting sequences of character IDs into meaningful embeddings.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTextEncoder(nn.Module):
    """
    Transformer encoder for Pokemon Red text.
    
    This encoder processes sequences of Pokemon Red character IDs and produces
    fixed-size embeddings that capture the semantic meaning of the text.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
        padding_idx: int = 0,
    ):
        """
        Initialize transformer text encoder.
        
        Args:
            vocab_size: Size of character vocabulary (256 for Pokemon Red)
            embed_dim: Dimension of character embeddings
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer encoder layers
            ff_dim: Dimension of feedforward network
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            output_dim: Output embedding dimension (if None, uses embed_dim)
            padding_idx: Index to use for padding (usually 0)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim if output_dim is not None else embed_dim
        self.padding_idx = padding_idx
        
        # Character embedding layer
        self.char_embedding = nn.Embedding(
            vocab_size, 
            embed_dim, 
            padding_idx=padding_idx
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection
        if self.output_dim != embed_dim:
            self.output_proj = nn.Linear(embed_dim, self.output_dim)
        else:
            self.output_proj = nn.Identity()
        
        # Layer normalization
        self.output_norm = nn.LayerNorm(self.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.char_embedding.weight, mean=0, std=0.02)
        if self.padding_idx is not None:
            nn.init.constant_(self.char_embedding.weight[self.padding_idx], 0)
        
        # Initialize output projection
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
    
    def create_padding_mask(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            text_ids: Tensor of character IDs, shape (batch_size, seq_len)
            
        Returns:
            Boolean mask, True for positions to mask (padding), shape (batch_size, seq_len)
        """
        return text_ids == self.padding_idx
    
    def forward(
        self, 
        text_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through text encoder.
        
        Args:
            text_ids: Character IDs, shape (batch_size, seq_len)
            mask: Optional attention mask, shape (batch_size, seq_len)
                  True for positions to mask (padding)
            return_sequence: If True, return full sequence embeddings.
                           If False, return pooled embedding (mean over sequence).
                           
        Returns:
            If return_sequence=False: Pooled text embedding, shape (batch_size, output_dim)
            If return_sequence=True: Sequence embeddings, shape (batch_size, seq_len, output_dim)
        """
        # Create padding mask if not provided
        if mask is None:
            mask = self.create_padding_mask(text_ids)
        
        # Embed characters: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embedded = self.char_embedding(text_ids)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Pass through transformer encoder
        # Note: PyTorch transformer uses inverted mask convention
        # True means "do not attend", so we pass mask directly
        encoded = self.transformer_encoder(
            embedded,
            src_key_padding_mask=mask
        )
        
        # Project to output dimension
        output = self.output_proj(encoded)
        output = self.output_norm(output)
        
        if return_sequence:
            # Return full sequence
            return output
        else:
            # Pool over sequence dimension (mean pooling, excluding padding)
            # Create mask for mean: (batch_size, seq_len, 1)
            pool_mask = (~mask).unsqueeze(-1).float()
            masked_output = output * pool_mask
            
            # Mean over non-padded positions
            sum_output = masked_output.sum(dim=1)
            count = pool_mask.sum(dim=1).clamp(min=1.0)  # Avoid division by zero
            pooled = sum_output / count
            
            return pooled
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """
        Load pretrained weights from checkpoint.
        
        This method is designed to support loading pretrained text encoders,
        potentially from pre-trained language models.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce that keys match
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        
        if not strict:
            print(f"Loaded pretrained text encoder from {checkpoint_path}")
            if missing_keys:
                print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
        
        return missing_keys, unexpected_keys
    
    def freeze_embeddings(self):
        """Freeze character embedding layer (useful for fine-tuning)."""
        self.char_embedding.weight.requires_grad = False
    
    def unfreeze_embeddings(self):
        """Unfreeze character embedding layer."""
        self.char_embedding.weight.requires_grad = True
    
    def freeze_transformer(self):
        """Freeze transformer encoder layers (useful for fine-tuning)."""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_transformer(self):
        """Unfreeze transformer encoder layers."""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True


class LightweightTextEncoder(nn.Module):
    """
    Lightweight text encoder using only embeddings and MLPs.
    
    This is a simpler alternative to the transformer encoder that may be
    more efficient for initial experiments or resource-constrained scenarios.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        """
        Initialize lightweight text encoder.
        
        Args:
            vocab_size: Size of character vocabulary
            embed_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden layers
            output_dim: Output embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            padding_idx: Index to use for padding
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        
        # Character embedding
        self.char_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx
        )
        
        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # MLP layers
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.char_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
        if self.padding_idx is not None:
            nn.init.constant_(self.char_embedding.weight[self.padding_idx], 0)
    
    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through lightweight encoder.
        
        Args:
            text_ids: Character IDs, shape (batch_size, seq_len)
            
        Returns:
            Text embedding, shape (batch_size, output_dim)
        """
        batch_size, seq_len = text_ids.shape
        
        # Embed characters
        char_emb = self.char_embedding(text_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=text_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        embedded = char_emb + pos_emb
        
        # Pool over sequence (mean pooling, excluding padding)
        mask = (text_ids != self.padding_idx).unsqueeze(-1).float()
        masked_emb = embedded * mask
        pooled = masked_emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        
        # Pass through MLP
        output = self.encoder(pooled)
        
        return output

