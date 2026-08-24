import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

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

class AttentionPooler(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True) # self.attn = ExplicitMultiheadAttention(dim, num_heads, dropout=0.0)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, Dim)
        batch_size = x.shape[0]
        q = self.query.expand(batch_size, -1, -1) # (B, 1, Dim)
        
        # Native SDPA requires True for valid and False for padding,
        # but nn.MultiheadAttention expects True for padding tokens!
        # key_padding_mask should be True for tokens to ignore.
        key_padding_mask = ~mask.bool() if mask is not None else None
        
        # Query attends to the sequence
        attn_out, _ = self.attn(
            q, x, x, 
            key_padding_mask=key_padding_mask
        )
        
        return self.norm(attn_out.squeeze(1)) # (B, Dim)
    
    
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

        # Separate unfused linear layers for Q, K, V and Output
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
        
        if need_weights:
            raise ValueError("need_weights=True is not supported by PyTorch's native F.scaled_dot_product_attention.")
            # NOTE: If you need weights, conditionally call your old custom implementation here instead.

        # 1. Prepare the Attention Mask
        attn_mask = None
        if key_padding_mask is not None:
            # Invert the mask: Native SDPA expects True for valid tokens and False for padding.
            # Then expand from (batch_size, seq_len) to (batch_size, 1, 1, seq_len)
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)

        # 2. Extract dropout probability depending on train/eval mode
        # (Assuming self.dropout is an nn.Dropout module)
        dropout_p = self.dropout.p if self.training else 0.0

        # 3. Native Scaled Dot-Product Attention.  V3's layer fusion reshapes
        # [B, tokens, layers, D] into [B*tokens, layers, D].  At long context
        # lengths this can make batch*heads exceed the CUDA SDPA kernel's grid
        # limit even though the attended sequence itself is only three items.
        # Use the equivalent eager formulation for that unusual regime.  The
        # ordinary V2/V3 sequence path remains on fused SDPA.
        sdpa_grid_limit = 65535
        if attn_mask is None and batch_size * self.num_heads > sdpa_grid_limit:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)
            if dropout_p:
                attn_probs = F.dropout(attn_probs, p=dropout_p, training=True)
            attn_output = torch.matmul(attn_probs, v)
        else:
            # It automatically scales queries by 1 / sqrt(head_dim).
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, None
    """
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
            return output, attn_weights
        else:
            return output, None
    """

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

        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x
        
class JinaToSDXLAdapterV2(nn.Module):
    """
    Adapter specifically designed for jina-clip-v2 to SDXL.
    Replaces Q-Former architecture with an MLP projection and optional self-attention,
    preserving the input sequence length for fully variable-length outputs.
    """
    def __init__(self,
                 llm_dim=1024,           
                 sdxl_seq_dim=2048,
                 sdxl_pooled_dim=1280,
                 n_attention_blocks=4,   # Set to 0 if you only want pure MLP projection
                 num_heads=16,
                 dropout=0,
                 max_seq_len=512): # 
        super().__init__()
        
        # 1. Sequence Projection (Variable Length: LLM Dim -> SDXL Seq Dim)
        # Deep MLP ensures sophisticated cross-channel mapping 
        self.seq_projection = nn.Sequential(
            nn.Linear(llm_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        )
        
        # Allows the attention blocks to map Jina token index 'X' to SDXL token index 'Y'
        # self.positional_embedding = nn.Embedding(max_seq_len, sdxl_seq_dim)

        # Optional Self-Attention blocks to allow tokens to communicate post-projection
        # Since Jina embeddings are heavily contextualized, a few blocks (or none) are enough
        self.attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(n_attention_blocks)
        ])

        # 2. Pooled Projection (1024 -> 1280)
        #self.pooled_projection = nn.Sequential(
        #    nn.Linear(llm_dim, llm_dim),
        #    nn.LayerNorm(llm_dim),
        #    nn.GELU(),
        #    nn.Dropout(dropout),
        #    nn.Linear(llm_dim, sdxl_pooled_dim)
        #)
        # 2048 -> 1280
        self.attention_pooler = AttentionPooler(sdxl_seq_dim)
        self.pooled_projection = nn.Linear(sdxl_seq_dim, sdxl_pooled_dim)


    def forward(self, jina_hidden_states, jina_mean_pooled_state, attention_mask=None):
        """
        jina_hidden_states: (Batch, Seq_Len, 1024)
        jina_mean_pooled_state: (Batch, 1024) - Generated via Jina's mean_pooling
        attention_mask: (Batch, Seq_Len) - 1 for valid tokens, 0 for padding
        """
        # --- 1. Sequence Processing (Preserves Seq_Len identically) ---
        # Map 1024 -> 2048 across the last dimension
        hidden_states = self.seq_projection(jina_hidden_states)

        # seq_len = hidden_states.size(1)
        # positions = torch.arange(seq_len, device=hidden_states.device)
        # hidden_states = hidden_states + self.positional_embedding(positions).unsqueeze(0)


        
        # Apply optional attention layers
        for block in self.attention_blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        pooled_features = self.attention_pooler(hidden_states, attention_mask)
        pooled_output = self.pooled_projection(pooled_features)


        return hidden_states, pooled_output

