import torch
import torch.nn as nn
import torch.nn.functional as F # Import functional
import logging

# This is different from the custom node code, since the multiheadattention layers weren't being found by the LoRA 
# Vibe-coded


logger = logging.getLogger("LLM-SDXL-Adapter")

def pad_to_length(tensor, target_length, dim=1, value=0):
    """Universal function for padding tensors"""
    current_length = tensor.size(dim)

    if current_length >= target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_size = list(tensor.shape)
    pad_size[dim] = target_length - current_length

    padding = torch.full(
        pad_size,
        value,
        device=tensor.device,
        dtype=tensor.dtype
    )

    return torch.cat([tensor, padding], dim=dim)


class ExplicitMultiheadAttention(nn.Module):
    """
    An explicit implementation of Multi-head Attention to ensure LoRA compatibility.
    Replaces the monolithic nn.MultiheadAttention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Separate linear layers for Q, K, V and Output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, average_attn_weights=True):
        batch_size, seq_len, _ = query.shape

        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len, head_dim)
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, key_padding_mask)

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)

        if need_weights:
            # Note: weight processing might differ slightly from nn.MultiheadAttention's output format
            return output, attn_weights
        else:
            return output, None

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            # The mask shape is (batch_size, seq_len)
            # We need to broadcast it to (batch_size, num_heads, seq_len, seq_len)
            # The mask is True for padded tokens, so we fill with -inf
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, v)
        return output, attn_probs


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        # FIX: Use the new explicit attention module
        self.attn = ExplicitMultiheadAttention(
            dim, num_heads, dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, mask=None):
        # Self-attention
        normed = self.norm1(x)

        if mask is not None:
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        # Call the new attention module
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class LLMToSDXLAdapter(nn.Module):
    """
    Universal adapter for converting any LLM embeddings to SDXL format
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """
    def __init__(self,
                 llm_dim=1152,
                 sdxl_seq_dim=2048,
                 sdxl_pooled_dim=1280,
                 max_input_len=512,
                 target_seq_len=308,
                 n_wide_blocks=3,
                 n_narrow_blocks=3,
                 num_heads=16,
                 dropout=0):
        super().__init__()

        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads

        # Projections
        # Use a more specific check to avoid creating an empty layer
        if llm_dim != sdxl_seq_dim:
            self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim)
        else:
            self.seq_projection = None # Explicitly set to None


        # Positional embeddings
        self.input_position_embeddings = nn.Parameter(
            torch.randn(1, max_input_len, sdxl_seq_dim)
        )
        self.output_position_embeddings = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        
        # Wide blocks
        self.wide_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_wide_blocks)
        ])

        # Compression: Cross-attention
        self.compression_queries = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        # FIX: Use the new explicit attention module
        self.compression_attention = ExplicitMultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        self.compression_gate = nn.Sequential(
            nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim),
            nn.Sigmoid()
        )

        # Narrow blocks
        self.narrow_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_narrow_blocks)
        ])
        
        # Pooling head
        # FIX: Use the new explicit attention module
        self.pooling_attention = ExplicitMultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.pooling_token = nn.Parameter(torch.randn(1, 1, sdxl_seq_dim))
        self.pooled_projection = nn.Sequential(
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_pooled_dim)
        )

    def forward(self, llm_hidden_states, attention_mask=None):
        batch_size, seq_len, _ = llm_hidden_states.shape

        if self.seq_projection is not None:
            hidden_states = self.seq_projection(llm_hidden_states)
        else:
            hidden_states = llm_hidden_states  

        if seq_len > self.max_input_len:
            hidden_states = hidden_states[:, :self.max_input_len, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_input_len]
        else:
            if seq_len < self.max_input_len:
                hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1)
                if attention_mask is not None:
                    attention_mask = pad_to_length(attention_mask, self.max_input_len, dim=1, value=0)
                else:
                    attention_mask = torch.ones(batch_size, self.max_input_len, device=hidden_states.device)
                    attention_mask[:, seq_len:] = 0

        hidden_states = hidden_states + self.input_position_embeddings

        for block in self.wide_attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        queries = self.compression_queries.expand(batch_size, -1, -1)
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        compressed_sequence, _ = self.compression_attention(
            queries,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask
        )

        gate_input = torch.cat([queries, compressed_sequence], dim=-1)
        gate_weights = self.compression_gate(gate_input)
        compressed_sequence = gate_weights * compressed_sequence + (1 - gate_weights) * queries
        compressed_sequence = self.compression_norm(compressed_sequence)
        compressed_sequence = compressed_sequence + self.output_position_embeddings

        for block in self.narrow_attention_blocks:
            compressed_sequence = block(compressed_sequence)

        pooling_tokens = self.pooling_token.expand(batch_size, -1, -1)
        pooled_output, _ = self.pooling_attention(
            pooling_tokens,
            compressed_sequence,
            compressed_sequence
        )
        pooled_output = pooled_output.squeeze(1)
        pooled_output = self.pooled_projection(pooled_output)

        return compressed_sequence, pooled_output