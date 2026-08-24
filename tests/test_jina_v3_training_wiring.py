import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from library.sdxl_original_unet import CrossAttention
from llm_adapter_lib.jina.jina_to_sdxl_adapter_v3 import JinaToSDXLAdapterV3


ROOT = Path(__file__).resolve().parents[1]


class _Logger:
    def info(self, *args, **kwargs):
        pass


def _load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_new_module_only_mode_freezes_v2_compatible_path():
    configure = _load_function(
        ROOT / "sdxl_train.py",
        "configure_jina_v3_new_modules_only",
        {"JinaToSDXLAdapterV3": JinaToSDXLAdapterV3},
    )
    adapter = JinaToSDXLAdapterV3(
        llm_dim=4,
        sdxl_seq_dim=8,
        sdxl_pooled_dim=6,
        n_attention_blocks=1,
        num_heads=2,
    )
    configure(adapter)
    assert all(not parameter.requires_grad for parameter in adapter.seq_projection.parameters())
    assert all(not parameter.requires_grad for parameter in adapter.attention_blocks.parameters())
    assert all(parameter.requires_grad for parameter in adapter.layer_fusion.parameters())
    assert all(parameter.requires_grad for parameter in adapter.mean_pooled_projection.parameters())


def test_cross_attention_mask_helper_returns_bool_and_validates_shape():
    helper = _load_function(
        ROOT / "sdxl_train.py",
        "prepare_jina_adapter_cross_attention_mask",
        {"torch": torch},
    )
    args = SimpleNamespace(adapter_jina=True, jina_adapter_cross_attn_mask=True)
    embedding = torch.randn(2, 5, 8)
    condition = {"attention_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])}
    mask = helper(args, condition, embedding, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.shape == embedding.shape[:2]


def test_gradient_checkpointing_marks_jina_conditioning_tensors_not_dict_keys():
    helper = _load_function(
        ROOT / "train_network.py",
        "_enable_text_conds_grad",
        {},
    )
    condition = {
        "prompt_embeds": torch.randn(2, 5, 8),
        "pooled_prompt_embeds": torch.randn(2, 6),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    helper(SimpleNamespace(use_llm_as_text_encoder=True), condition)
    assert condition["prompt_embeds"].requires_grad
    assert condition["pooled_prompt_embeds"].requires_grad
    assert not condition["attention_mask"].requires_grad


def test_native_eager_and_sdpa_cross_attention_ignore_masked_key_values():
    hidden = torch.randn(1, 3, 4)
    context = torch.randn(1, 5, 4)
    perturbed = context.clone()
    perturbed[:, 3:] += 10000
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    for use_sdpa in (False, True):
        attention = CrossAttention(query_dim=4, cross_attention_dim=4, heads=2, dim_head=2).eval()
        attention.set_use_sdpa(use_sdpa)
        with torch.no_grad():
            first = attention(hidden, context, mask)
            second = attention(hidden, perturbed, mask)
        torch.testing.assert_close(first, second)


def test_cli_exposes_v3_without_component_learning_rate_options():
    sdxl_source = (ROOT / "sdxl_train.py").read_text(encoding="utf-8-sig")
    network_source = (ROOT / "train_network.py").read_text(encoding="utf-8-sig")
    inherited_source = (ROOT / "sdxl_train_network.py").read_text(encoding="utf-8-sig")
    train_util_source = (ROOT / "library" / "train_util.py").read_text(encoding="utf-8-sig")
    assert '"--train_jina_v3_new_modules_only"' in sdxl_source
    assert "hidden_state_layer_indices=required_hidden_state_layers" in sdxl_source
    assert "hidden_state_layer_indices=required_hidden_state_layers" in network_source
    assert "missing_or_mismatched_v3_keys(adapter, checkpoint_state)" in network_source
    assert "layer_fusion\\.attention_blocks" in network_source
    assert "parser = train_network.setup_parser()" in inherited_source
    assert "choices=[None, 150, 225, 300, 512, 1024, 2048]" in train_util_source
