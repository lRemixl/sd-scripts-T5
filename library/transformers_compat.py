"""Compatibility helpers for supported Transformers 4.x and 5.x releases."""

from contextlib import contextmanager
import json

from packaging.version import Version
import transformers
from transformers import CLIPTextConfig
from transformers import CLIPTextModel as _NativeCLIPTextModel


TRANSFORMERS_VERSION = Version(transformers.__version__)
JINA_CLIP_FIX_MISTRAL_REGEX = False
JINA_CLIP_TRUST_REMOTE_CODE = True


def validate_jina_clip_tokenizer(tokenizer):
    """Reject Transformers' Mistral pre-tokenizer patch for Jina XLM-R.

    jina-clip-v2's official tokenizer is an XLM-R Unigram tokenizer whose
    pre-tokenizer is ``WhitespaceSplit`` followed by ``Metaspace``. Some
    Transformers releases incorrectly classify a local Jina model directory
    as Mistral and recommend ``fix_mistral_regex=True``. That option prepends a
    Mistral ``Split`` regex and changes Jina token IDs.

    Return a short description when the tokenizers backend exposes its state.
    Older slow-tokenizer implementations do not, so the explicit loader flag
    remains the authoritative safeguard there.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    get_state = getattr(pre_tokenizer, "__getstate__", None)
    if not callable(get_state):
        return "backend pre-tokenizer unavailable"

    state = get_state()
    if isinstance(state, bytes):
        state = state.decode("utf-8")
    if isinstance(state, str):
        state = json.loads(state)
    if not isinstance(state, dict):
        return "backend pre-tokenizer unavailable"

    components = state.get("pretokenizers", []) if state.get("type") == "Sequence" else [state]
    component_types = tuple(component.get("type") for component in components if isinstance(component, dict))
    if component_types and component_types[0] == "Split":
        raise RuntimeError(
            "jina-clip-v2 was loaded with Transformers' Mistral regex repair. "
            "That repair is not part of Jina's XLM-R tokenizer and changes token IDs. "
            "Reload with fix_mistral_regex=False."
        )
    if component_types[:2] == ("WhitespaceSplit", "Metaspace"):
        return "WhitespaceSplit+Metaspace"
    return "+".join(component_types) or "backend pre-tokenizer unavailable"


def load_jina_clip_tokenizer(
    model_name_or_path,
    *,
    local_files_only=False,
    revision=None,
    **kwargs,
):
    """Load the tokenizer used to pretrain jina-clip-v2's text tower."""
    from transformers import AutoTokenizer

    if kwargs.pop("fix_mistral_regex", JINA_CLIP_FIX_MISTRAL_REGEX):
        raise ValueError("jina-clip-v2 must not use the Mistral tokenizer regex repair.")
    load_kwargs = {
        "trust_remote_code": JINA_CLIP_TRUST_REMOTE_CODE,
        "fix_mistral_regex": JINA_CLIP_FIX_MISTRAL_REGEX,
        "local_files_only": local_files_only,
        **kwargs,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
    validate_jina_clip_tokenizer(tokenizer)
    return tokenizer


@contextmanager
def jina_clip_load_context(local_files_only=False):
    """Constrain tokenizer and Hub lookups made inside Jina's trusted code.

    The jina-clip-v2 implementation creates its Jina-XLM-R text tower through
    ``AutoModel.from_config`` and constructs another tokenizer inside that
    tower. Pin the nested tokenizer to Jina's native pre-tokenizer. For local
    model directories, also propagate ``local_files_only`` to the dynamic-code
    resolver because Transformers does not inherit the outer value.
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer
    from transformers.models.auto import auto_factory

    auto_classes = (AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer)
    previous_pretrained = []
    for auto_class in auto_classes:
        had_override = "from_pretrained" in auto_class.__dict__
        descriptor = auto_class.__dict__.get("from_pretrained")
        original = auto_class.from_pretrained

        def constrained_from_pretrained(
            cls,
            *args,
            _original=original,
            _is_tokenizer=auto_class is AutoTokenizer,
            **kwargs,
        ):
            if local_files_only:
                kwargs["local_files_only"] = True
            if _is_tokenizer:
                kwargs["fix_mistral_regex"] = JINA_CLIP_FIX_MISTRAL_REGEX
            return _original(*args, **kwargs)

        previous_pretrained.append((auto_class, had_override, descriptor))
        auto_class.from_pretrained = classmethod(constrained_from_pretrained)

    # Jina's HFTextEncoder always passes code_revision to AutoModel.from_config,
    # including when its value is None. Some Transformers releases forward that
    # loader-only None into an already registered XLMRobertaLoRA constructor.
    # Preserve real revision pins, but discard the meaningless None before it
    # can reach XLMRobertaLoRA.__init__.
    had_from_config_override = "from_config" in AutoModel.__dict__
    previous_from_config = AutoModel.__dict__.get("from_config")
    original_from_config = AutoModel.from_config

    def compatible_from_config(cls, config, **kwargs):
        if kwargs.get("code_revision") is None:
            kwargs.pop("code_revision", None)
        return original_from_config(config, **kwargs)

    AutoModel.from_config = classmethod(compatible_from_config)

    # AutoModel.from_config forwards its kwargs both to the dynamic-code
    # resolver and to the eventual model constructor in current Transformers.
    # Injecting local_files_only into from_config would likewise reach the
    # model's __init__, so patch only the resolver symbol imported by
    # auto_factory for the offline constraint.
    original_dynamic_loader = auto_factory.get_class_from_dynamic_module

    def local_dynamic_loader(*args, **kwargs):
        kwargs["local_files_only"] = True
        return original_dynamic_loader(*args, **kwargs)

    if local_files_only:
        auto_factory.get_class_from_dynamic_module = local_dynamic_loader
    try:
        yield
    finally:
        if local_files_only:
            auto_factory.get_class_from_dynamic_module = original_dynamic_loader
        if had_from_config_override:
            AutoModel.from_config = previous_from_config
        else:
            delattr(AutoModel, "from_config")
        for auto_class, had_override, descriptor in reversed(previous_pretrained):
            if had_override:
                auto_class.from_pretrained = descriptor
            else:
                delattr(auto_class, "from_pretrained")


def pretrained_dtype_kwargs(dtype):
    """Return the version-correct dtype keyword for ``from_pretrained``.

    Transformers 5 renamed ``torch_dtype`` to ``dtype``. Passing a
    ``torch.dtype`` through the deprecated keyword reaches some trusted remote
    config classes unchanged; Jina CLIP v2 then treats it as an attribute name
    and raises ``TypeError: attribute name must be string``.
    """
    if TRANSFORMERS_VERSION.major >= 5:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def ensure_legacy_clip_symbols():
    """Restore small CLIP helpers expected by trusted remote model code.

    Jina CLIP v2 imports ``clip_loss`` from Transformers' private CLIP module.
    Transformers 5 removed that helper even though the output classes used by
    Jina remain available. Recreate the original two-way contrastive loss so
    the pinned Jina implementation can still be loaded through ``AutoModel``.
    """
    from transformers.models.clip import modeling_clip

    if not hasattr(modeling_clip, "clip_loss"):
        import torch
        import torch.nn.functional as F

        def contrastive_loss(logits):
            return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

        def clip_loss(similarity):
            caption_loss = contrastive_loss(similarity)
            image_loss = contrastive_loss(similarity.t())
            return (caption_loss + image_loss) / 2.0

        modeling_clip.clip_loss = clip_loss


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _jina_eva_rotary_frequencies(model, freqs_cos, freqs_sin):
    """Recreate EVA02's 2-D rotary tables without importing remote code."""
    import math
    import torch

    if freqs_cos.ndim != 2 or freqs_sin.shape != freqs_cos.shape:
        raise RuntimeError(
            "Jina EVA rotary buffer shape mismatch while repairing Transformers compatibility: "
            f"cos={tuple(freqs_cos.shape)}, sin={tuple(freqs_sin.shape)}"
        )

    position_count, rotary_width = freqs_cos.shape
    ft_seq_len = math.isqrt(position_count)
    if ft_seq_len * ft_seq_len != position_count or rotary_width % 4:
        raise RuntimeError(
            "Jina EVA rotary buffers have an unsupported geometry: "
            f"shape={tuple(freqs_cos.shape)}"
        )

    model_config = getattr(model, "config", None)
    vision_config = _config_value(model_config, "vision_config")
    pt_seq_len = int(_config_value(vision_config, "pt_hw_seq_len", 16))
    if pt_seq_len <= 0:
        raise RuntimeError(
            f"Jina EVA pt_hw_seq_len must be positive, got {pt_seq_len}."
        )

    # Mirrors jina-clip-implementation/rope_embeddings.py exactly. The remote
    # EVA builder uses the default language frequencies and theta=10000, then
    # interpolates the pretraining grid to the current square patch grid.
    rotary_dim = rotary_width // 2
    device = freqs_cos.device
    base_frequencies = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    positions = (
        torch.arange(ft_seq_len, device=device, dtype=torch.float32)
        / ft_seq_len
        * pt_seq_len
    )
    axis_phases = torch.einsum("i,j->ij", positions, base_frequencies)
    axis_phases = axis_phases.repeat_interleave(2, dim=-1)
    phases = torch.cat(
        (
            axis_phases[:, None, :].expand(ft_seq_len, ft_seq_len, rotary_dim),
            axis_phases[None, :, :].expand(ft_seq_len, ft_seq_len, rotary_dim),
        ),
        dim=-1,
    ).reshape(position_count, rotary_width)
    return (
        phases.cos().to(dtype=freqs_cos.dtype),
        phases.sin().to(dtype=freqs_sin.dtype),
    )


def repair_jina_clip_nonpersistent_buffers(model):
    """Restore Jina remote-code buffers skipped by Transformers 5 loading.

    Jina CLIP v2's trusted remote model creates text rotary frequencies, LoRA
    dropout masks, and EVA02 vision rotary tables in its constructors.
    Transformers 5 builds the model under its no-initialization context, so
    these checkpoint-omitted buffers can be left as uninitialized memory.

    Recomputing the frequencies and resetting the masks to their constructor
    value is safe on Transformers 4 as well, and makes both major versions
    produce the same encoder outputs for the same checkpoint.
    """
    import torch

    masks_seen = 0
    rotary_seen = 0
    eva_rotary_seen = 0
    repaired = 0
    vision_model = getattr(model, "vision_model", None)
    vision_rope = getattr(vision_model, "rope", None)

    with torch.no_grad():
        for module in model.modules():
            dropout_mask = module._buffers.get("lora_dropout_mask")
            if isinstance(dropout_mask, torch.Tensor):
                masks_seen += 1
                if not bool(torch.eq(dropout_mask, 1).all()):
                    dropout_mask.fill_(1)
                    repaired += 1

            inv_freq = module._buffers.get("inv_freq")
            compute_inv_freq = getattr(module, "_compute_inv_freq", None)
            if isinstance(inv_freq, torch.Tensor) and callable(compute_inv_freq):
                rotary_seen += 1
                expected = compute_inv_freq(device=inv_freq.device).to(dtype=inv_freq.dtype)
                if expected.shape != inv_freq.shape:
                    raise RuntimeError(
                        "Jina rotary buffer shape mismatch while repairing Transformers compatibility: "
                        f"loaded={tuple(inv_freq.shape)}, expected={tuple(expected.shape)}"
                    )
                if not torch.equal(inv_freq, expected):
                    inv_freq.copy_(expected)
                    repaired += 1
                    # No forward should have run yet, but invalidate cached
                    # trigonometric tables defensively if the remote class has them.
                    if hasattr(module, "_cos_cached"):
                        module._cos_cached = None
                    if hasattr(module, "_sin_cached"):
                        module._sin_cached = None
                    if hasattr(module, "_cos_k_cached"):
                        module._cos_k_cached = None
                    if hasattr(module, "_sin_k_cached"):
                        module._sin_k_cached = None
                    if hasattr(module, "_seq_len_cached"):
                        module._seq_len_cached = 0

            freqs_cos = module._buffers.get("freqs_cos")
            freqs_sin = module._buffers.get("freqs_sin")
            is_eva_rotary = (
                module is vision_rope
                or module.__class__.__name__ in {
                    "VisionRotaryEmbedding",
                    "VisionRotaryEmbeddingFast",
                }
            )
            if (
                is_eva_rotary
                and isinstance(freqs_cos, torch.Tensor)
                and isinstance(freqs_sin, torch.Tensor)
            ):
                eva_rotary_seen += 2
                expected_cos, expected_sin = _jina_eva_rotary_frequencies(
                    model,
                    freqs_cos,
                    freqs_sin,
                )
                if not torch.equal(freqs_cos, expected_cos):
                    freqs_cos.copy_(expected_cos)
                    repaired += 1
                if not torch.equal(freqs_sin, expected_sin):
                    freqs_sin.copy_(expected_sin)
                    repaired += 1

    return {
        "lora_dropout_masks": masks_seen,
        "rotary_inv_freq": rotary_seen,
        "eva_rotary_freqs": eva_rotary_seen,
        "repaired": repaired,
    }


if TRANSFORMERS_VERSION.major < 5:
    CLIPTextModel = _NativeCLIPTextModel
else:
    from transformers.models.clip.modeling_clip import CLIPPreTrainedModel

    class CLIPTextModel(CLIPPreTrainedModel):
        """Transformers 5 CLIP text model with the Transformers 4 module layout.

        Transformers 5 flattened ``CLIPTextModel.text_model`` into the model
        itself. Stable Diffusion checkpoints and this codebase use keys such as
        ``text_model.encoder.layers.*`` and access ``model.text_model`` during
        training. Wrapping the native 5.x implementation preserves that public
        contract without copying any CLIP implementation details.
        """

        config_class = CLIPTextConfig
        _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]
        _supports_flash_attn = False

        def __init__(self, config: CLIPTextConfig):
            super().__init__(config)
            # The native 5.x model initializes its own parameters in post_init().
            # Do not call post_init() again on this wrapper.
            self.text_model = _NativeCLIPTextModel._from_config(config)

        def get_input_embeddings(self):
            return self.text_model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.text_model.set_input_embeddings(value)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            **kwargs,
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )


__all__ = [
    "CLIPTextModel",
    "JINA_CLIP_FIX_MISTRAL_REGEX",
    "JINA_CLIP_TRUST_REMOTE_CODE",
    "TRANSFORMERS_VERSION",
    "ensure_legacy_clip_symbols",
    "jina_clip_load_context",
    "load_jina_clip_tokenizer",
    "pretrained_dtype_kwargs",
    "repair_jina_clip_nonpersistent_buffers",
    "validate_jina_clip_tokenizer",
]

