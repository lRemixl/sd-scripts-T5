"""Jina CLIP-v2 to SDXL adapter with token-wise layer fusion.

V3 deliberately keeps V2's sequence projection and four sequence transformer
blocks under the same names.  A zero-initialized residual gate makes the V3
sequence path start from Jina's final encoder output, so compatible V2 weights
remain an exact warm start.
"""

from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from .jina_to_sdxl_adapter_v2 import TransformerBlock


class LayerwiseAttentionFusion(nn.Module):
    """Fuse several representations of each token without mixing token positions."""

    def __init__(
        self,
        dim: int = 1024,
        num_layers: int = 3,
        num_blocks: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("LayerwiseAttentionFusion requires at least two layers.")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.input_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.depth_embeddings = nn.Parameter(torch.zeros(1, num_layers, dim))
        nn.init.normal_(self.depth_embeddings, mean=0.0, std=0.02)
        self.attention_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_score = nn.Linear(dim, 1)
        self.output_projection = nn.Linear(dim, dim)

        # Only the gate is zero initialized.  Zeroing the delta projection too
        # would prevent the fusion stack from receiving gradients after the gate
        # begins to move.
        self.channel_gate = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 4:
            raise ValueError(
                "LayerwiseAttentionFusion expects [batch, layers, sequence, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        batch, layers, sequence, width = hidden_states.shape
        if layers != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} selected layers, got {layers}.")
        if width != self.dim:
            raise ValueError(f"Expected hidden width {self.dim}, got {width}.")

        normalized = torch.stack(
            [self.input_norms[index](hidden_states[:, index]) for index in range(layers)],
            dim=1,
        )
        # [B, layers, tokens, D] -> [B*tokens, layers, D].  Each attention
        # operation therefore sees only the same token across encoder depth.
        layer_tokens = normalized.permute(0, 2, 1, 3).reshape(batch * sequence, layers, width)
        layer_tokens = layer_tokens + self.depth_embeddings.to(
            device=layer_tokens.device,
            dtype=layer_tokens.dtype,
        )
        for block in self.attention_blocks:
            layer_tokens = block(layer_tokens)

        scores = self.layer_score(layer_tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(layer_tokens * weights.unsqueeze(-1), dim=1)
        delta = self.output_projection(pooled).reshape(batch, sequence, width)
        weights = weights.reshape(batch, sequence, layers)
        return delta, weights

    def gated_residual(self, final_hidden_state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        gate = torch.tanh(self.channel_gate).to(
            device=delta.device,
            dtype=delta.dtype,
        )
        return final_hidden_state + delta * gate.view(1, 1, -1)


class JinaToSDXLAdapterV3(nn.Module):
    """SDXL adapter using token-wise attention over Jina layers h8, h16 and h24."""

    required_hidden_state_layers = (8, 16, 24)
    hidden_state_input_key = "jina_hidden_states_selected_layers"
    adapter_version = "v3"
    architecture_revision = 2

    def __init__(
        self,
        llm_dim: int = 1024,
        sdxl_seq_dim: int = 2048,
        sdxl_pooled_dim: int = 1280,
        n_attention_blocks: int = 4,
        num_heads: int = 16,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        layer_mix_init: str = "uniform",
    ):
        super().__init__()
        # layer_mix_init is retained as a deprecated, intentionally unused
        # argument so existing training configs continue to parse.
        del layer_mix_init
        self.max_seq_len = int(max_seq_len)
        self.llm_dim = int(llm_dim)
        self.num_selected_layers = len(self.required_hidden_state_layers)

        self.layer_fusion = LayerwiseAttentionFusion(
            dim=llm_dim,
            num_layers=self.num_selected_layers,
            num_blocks=2,
            num_heads=num_heads,
            mlp_ratio=2.0,
            dropout=dropout,
        )

        # Names and shapes intentionally match V2.
        self.seq_projection = nn.Sequential(
            nn.Linear(llm_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
        )
        self.attention_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    sdxl_seq_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(n_attention_blocks)
            ]
        )

        # Consumes Jina's official, pre-retrieval-normalization text feature.
        self.mean_pooled_projection = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, sdxl_pooled_dim),
        )

    @staticmethod
    def _validated_mask(
        attention_mask: torch.Tensor,
        expected_shape: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2 or tuple(attention_mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"attention_mask must have shape {tuple(expected_shape)}, "
                f"got {tuple(attention_mask.shape)}."
            )
        return attention_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _zero_padding(hidden_states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask is None:
            return hidden_states
        return hidden_states.masked_fill(~valid_mask.unsqueeze(-1), 0)

    def forward(
        self,
        jina_hidden_states_selected_layers: torch.Tensor,
        jina_mean_pooled_state: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected = jina_hidden_states_selected_layers
        if selected.dim() != 4:
            raise ValueError(
                "jina_hidden_states_selected_layers must be "
                "[batch, 3, sequence, 1024], "
                f"got {tuple(selected.shape)}."
            )
        if selected.shape[1] != self.num_selected_layers:
            raise ValueError(
                f"Expected h8/h16/h24 ({self.num_selected_layers} layers), got {selected.shape[1]}."
            )
        if selected.shape[-1] != self.llm_dim:
            raise ValueError(f"Expected Jina hidden size {self.llm_dim}, got {selected.shape[-1]}.")
        if jina_mean_pooled_state.dim() != 2:
            raise ValueError(
                "jina_mean_pooled_state must be [batch, 1024], "
                f"got {tuple(jina_mean_pooled_state.shape)}."
            )
        if jina_mean_pooled_state.shape[0] != selected.shape[0]:
            raise ValueError("Sequence and pooled Jina states have different batch sizes.")
        if jina_mean_pooled_state.shape[-1] != self.llm_dim:
            raise ValueError(
                f"Expected pooled Jina hidden size {self.llm_dim}, "
                f"got {jina_mean_pooled_state.shape[-1]}."
            )

        valid_mask = self._validated_mask(
            attention_mask,
            (selected.shape[0], selected.shape[2]),
            selected.device,
        )
        delta, _weights = self.layer_fusion(selected)
        fused = self.layer_fusion.gated_residual(selected[:, -1], delta)
        fused = self._zero_padding(fused, valid_mask)

        hidden_states = self.seq_projection(fused)
        hidden_states = self._zero_padding(hidden_states, valid_mask)
        for block in self.attention_blocks:
            hidden_states = block(hidden_states, valid_mask)
            hidden_states = self._zero_padding(hidden_states, valid_mask)

        pooled_output = self.mean_pooled_projection(jina_mean_pooled_state)
        return hidden_states, pooled_output


def filter_compatible_adapter_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """Keep checkpoint tensors whose names and shapes match ``module``."""

    target_state = module.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    unexpected: List[str] = []
    shape_mismatches: List[str] = []

    for key, value in state_dict.items():
        target = target_state.get(key)
        if target is None:
            unexpected.append(key)
        elif tuple(target.shape) != tuple(value.shape):
            shape_mismatches.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(target.shape)}"
            )
        else:
            compatible[key] = value

    return compatible, unexpected, shape_mismatches


def missing_or_mismatched_v3_keys(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> List[str]:
    """Return every required redesigned-V3 tensor absent or shape-incompatible."""

    missing = []
    for key, target in module.state_dict().items():
        value = state_dict.get(key)
        if value is None:
            missing.append(key)
        elif tuple(value.shape) != tuple(target.shape):
            missing.append(key)
    return missing

