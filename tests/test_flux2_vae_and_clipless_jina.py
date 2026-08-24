import ast
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_UTIL_PATH = ROOT / "library" / "custom_sdxl_utils.py"
MODEL_UTIL_PATH = ROOT / "library" / "sdxl_model_util.py"


def _load_functions(path, function_names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _load_custom_vae_contract():
    return _load_functions(
        CUSTOM_UTIL_PATH,
        {
            "normalize_vae_type",
            "is_flux2_vae_type",
            "move_custom_vae_to_device",
            "_load_autoencoder_kl_flux2_cls",
            "_load_autoencoder_dc_cls",
            "load_custom_vae",
            "load_sdxl_text_encoders_from_checkpoint",
        },
        {
            "AutoencoderKL": object,
            "model_util": SimpleNamespace(),
            "os": os,
            "torch": torch,
        },
    )


class _FakeFlux2VAE(nn.Module):
    from_pretrained_calls = []

    def __init__(self, dtype):
        super().__init__()
        self.regular = nn.Linear(2, 2).to(dtype=dtype)
        self.sensitive = nn.Linear(2, 2).to(dtype=torch.float32)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.from_pretrained_calls.append((path, kwargs))
        return cls(kwargs["torch_dtype"])


def test_flux2_directory_load_preserves_diffusers_fp32_modules(tmp_path):
    contract = _load_custom_vae_contract()
    contract["_load_autoencoder_kl_flux2_cls"] = lambda: _FakeFlux2VAE
    _FakeFlux2VAE.from_pretrained_calls.clear()
    vae = contract["load_custom_vae"](
        str(tmp_path),
        "flux2",
        torch.bfloat16,
        torch.device("cpu"),
    )

    assert _FakeFlux2VAE.from_pretrained_calls[0][1]["torch_dtype"] is torch.bfloat16
    assert vae.regular.weight.dtype is torch.bfloat16
    assert vae.sensitive.weight.dtype is torch.float32


def test_flux2_device_move_does_not_flatten_mixed_dtypes():
    contract = _load_custom_vae_contract()
    vae = _FakeFlux2VAE(torch.bfloat16)
    contract["move_custom_vae_to_device"](
        vae,
        torch.device("cpu"),
        torch.bfloat16,
        "flux2",
    )
    assert vae.regular.weight.dtype is torch.bfloat16
    assert vae.sensitive.weight.dtype is torch.float32


def test_external_conditioning_skips_both_sdxl_clip_towers():
    contract = _load_custom_vae_contract()
    state_dict = {
        "conditioner.embedders.0.transformer.fake": torch.ones(1),
        "conditioner.embedders.1.model.fake": torch.ones(1),
    }
    loaded = contract["load_sdxl_text_encoders_from_checkpoint"](
        state_dict,
        torch.device("cpu"),
        load_text_encoders=False,
    )
    assert loaded == (None, None, None)


def test_clipless_sdxl_checkpoint_save_omits_clip_keys(tmp_path):
    contract = _load_functions(
        MODEL_UTIL_PATH,
        {"save_stable_diffusion_checkpoint"},
        {
            "convert_text_encoder_2_state_dict_to_sdxl": lambda state, scale: state,
            "model_util": SimpleNamespace(
                convert_vae_state_dict=lambda state: state,
                is_safetensors=lambda path: False,
            ),
            "save_file": lambda *args, **kwargs: None,
            "torch": torch,
        },
    )
    output_path = tmp_path / "clipless.ckpt"
    contract["save_stable_diffusion_checkpoint"](
        str(output_path),
        None,
        None,
        nn.Linear(2, 2),
        1,
        2,
        None,
        nn.Linear(2, 2),
        None,
        {},
    )
    state_dict = torch.load(output_path, map_location="cpu")["state_dict"]
    assert any(key.startswith("model.diffusion_model.") for key in state_dict)
    assert any(key.startswith("first_stage_model.") for key in state_dict)
    assert not any(key.startswith("conditioner.embedders.") for key in state_dict)


def test_jina_entrypoints_request_clipless_custom_loading():
    assert "load_text_encoders=not use_jina" in (ROOT / "sdxl_train.py").read_text(encoding="utf-8-sig")
    source = (ROOT / "sdxl_train_network.py").read_text(encoding="utf-8-sig")
    assert "load_text_encoders=not bool(" in source
    assert 'getattr(args, "use_llm_as_text_encoder", False)' in source
    assert 'getattr(args, "adapter_jina", False)' in source
