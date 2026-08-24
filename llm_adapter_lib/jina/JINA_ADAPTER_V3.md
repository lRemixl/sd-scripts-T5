# Jina SDXL adapter V3

V3 is opt-in. V2 remains the default and its implementation/checkpoint format
are unchanged.

## Architecture

The text wrapper captures Jina CLIP v2 layers h8, h16 and h24. h24 is taken
from the encoder's actual final sequence output so the zero-gated path exactly
matches the state consumed by V2.

```text
[h8, h16, h24] per token
  -> independent LayerNorm + learned depth embeddings
  -> two attention/MLP blocks across encoder depth only
  -> content-dependent softmax pooling across the three layers
  -> 1024-D delta projection

h24 + tanh(channel_gate) * delta
  -> V2-compatible seq_projection
  -> four V2-compatible sequence attention blocks
  -> SDXL cross-attention conditioning

Jina get_text_features (raw, before retrieval L2 normalization)
  -> 1024 -> 1024 -> 1280 mean_pooled_projection
  -> SDXL vector conditioning
```

The channel gate starts at zero. A V2 checkpoint therefore supplies an exact
sequence warm start on valid tokens. The new fusion and pooled modules still
need alignment training.

## Jina tokenizer and trusted architecture loading

Jina CLIP v2 uses the Jina-XLM-R tokenizer inherited from
`jina-embeddings-v3`. Keep `fix_mistral_regex=False`: the Transformers option
is a repair for a Mistral tokenizer regex, not for Jina's
`WhitespaceSplit`/`Metaspace` XLM-R pre-tokenizer. Enabling it changes token
boundaries and is a training/inference mismatch with existing V2/V3 adapters.

The loader passes `trust_remote_code=True` on every Jina model load because
the checkpoint selects custom architecture code through `auto_map`. This is an
authorization flag for each load, not a one-time compilation setting. A local
`--llm_model_path` is treated as fully offline, including the nested
`jina-embeddings-v3` and XLM-R implementation lookups. Cache those trusted
modules once before using a local-only path. For Hub loading, optional
`--llm_model_revision` and `--llm_code_revision` values pin the selected model
and outer custom-code revisions.

## Training with `sdxl_train.py`

```text
--use_llm_as_text_encoder
--adapter_jina
--jina_adapter_version v3
--adapter_learning_rate <learning-rate>
```

For a V2 warm start, also pass `--llm_adapter_path`. Compatible
`seq_projection` and `attention_blocks` tensors load; the V2 attention pooler is
ignored. Legacy scalar-mix V3 checkpoints also load compatible sequence and
pooled tensors while their `layer_mix.*` tensors are ignored.

Set an explicit positive `--max_token_length` (normally 512 or 1024). Jina
truncates to that length and then pads the batch to the nearest multiple of 77.

To train only the redesigned modules:

```text
--train_jina_v3_new_modules_only
--llm_adapter_path <v2-or-v3-warm-start.safetensors>
```

This freezes the V2-compatible sequence path and trains `layer_fusion` plus
`mean_pooled_projection`. It cannot be combined with other selective adapter
training modes. `--jina_layer_mix_init` is accepted only for old config
compatibility and has no effect.

This port intentionally uses one `--adapter_learning_rate` for every trainable
adapter parameter. The rough source tree's component-specific V3 learning-rate
flags are not part of this repository. In new-module-only mode, the same single
rate applies to `layer_fusion` and `mean_pooled_projection` while the inherited
path is frozen.

## Network / PEFT training

`train_network.py` and `sdxl_train_network.py` require a complete redesigned V3
base checkpoint because their base adapter is frozen. The built-in LyCORIS
targets include the fusion attention projections, fusion MLP linears, layer
score/output projection, sequence mapper, and pooled MLP.

The standalone `layer_fusion.channel_gate`, depth embeddings, and normalization
parameters are not ordinary LoRA targets. Train the base V3 with
`sdxl_train.py` before PEFT training.

## Masking

The adapter's optional `[B, L]` mask is used as a key-padding mask in every
sequence self-attention block. Padded query/output positions are explicitly
zeroed before and after the blocks. For the U-Net, enable
`--jina_adapter_cross_attn_mask` as well; zero-valued conditioning tokens still
consume cross-attention probability unless keys are masked. The native eager,
memory-efficient, and SDPA paths accept this mask. The xFormers path does not,
so the training entry points reject that combination rather than silently
ignoring the mask.
