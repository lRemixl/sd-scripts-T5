import ast
import hashlib
import re
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_UTIL_PATH = ROOT / "library" / "train_util.py"


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _load_nodes(names, namespace):
    tree = ast.parse(TRAIN_UTIL_PATH.read_text(encoding="utf-8-sig"), filename=str(TRAIN_UTIL_PATH))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef)) and node.name in names
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(TRAIN_UTIL_PATH), "exec"), namespace)
    return namespace


def _load_artist_contract():
    return _load_nodes(
        {"ImageInfo", "extract_balance_artist_tags", "apply_artist_balance_repeats"},
        {
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "Sequence": Sequence,
            "hashlib": hashlib,
            "logger": _Logger(),
            "re": re,
            "torch": torch,
            "unicodedata": unicodedata,
        },
    )


def test_artist_balance_repeats_equalizes_rare_artist_effective_count():
    contract = _load_artist_contract()
    image_info = contract["ImageInfo"]
    balance = contract["apply_artist_balance_repeats"]
    infos = [
        image_info("common-1", 2, "tag, @ Common", False, "common-1.png"),
        image_info("common-2", 2, "tag, @ common", False, "common-2.png"),
        image_info("common-3", 2, "tag, @ COMMON", False, "common-3.png"),
        image_info("rare", 2, "tag, @ Rare", False, "rare.png"),
    ]
    subset = SimpleNamespace(
        artist_balance_repeats=True,
        artist_balance_summary=None,
        caption_separator=",",
        image_dir="images",
        num_repeats=2,
    )

    summary = balance(infos, subset)

    assert [info.num_repeats for info in infos[:3]] == [2, 2, 2]
    assert infos[3].num_repeats == 6
    assert summary["effective_min"] == summary["effective_max"] == 6
    assert subset.artist_balance_summary == summary


def _load_flow_contract():
    class _CustomTrainFunctions:
        @staticmethod
        def apply_noise_offset(*args, **kwargs):
            raise AssertionError("noise offset should be disabled in this test")

        @staticmethod
        def pyramid_noise_like(*args, **kwargs):
            raise AssertionError("multires noise should be disabled in this test")

    def get_timesteps_and_huber_c(*args, timesteps_override=None, **kwargs):
        return timesteps_override, None

    return _load_nodes(
        {"flow_sigmas_to_timesteps", "get_flow_huber_c", "get_noise_noisy_latents_and_timesteps"},
        {
            "Optional": Optional,
            "cosine_optimal_transport": None,
            "custom_train_functions": _CustomTrainFunctions,
            "get_timesteps_and_huber_c": get_timesteps_and_huber_c,
            "torch": torch,
        },
    )


def _flow_args(continuous):
    return SimpleNamespace(
        adaptive_noise_scale=None,
        flow_continuous_timesteps=continuous,
        flow_logit_mean=0.0,
        flow_logit_std=1.0,
        flow_model=True,
        flow_timestep_distribution="uniform",
        flow_uniform_shift=False,
        flow_uniform_static_ratio=None,
        flow_use_ot=False,
        loss_type="l2",
        multires_noise_discount=0.3,
        multires_noise_iterations=0,
        noise_offset=0.0,
        noise_offset_random_strength=False,
    )


def test_continuous_flow_timestep_conditioning_is_opt_in_and_matches_noising_sigma():
    contract = _load_flow_contract()
    get_noise = contract["get_noise_noisy_latents_and_timesteps"]
    scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    latents = torch.zeros(4, 1, 1, 1)

    torch.manual_seed(123)
    noise, noisy_latents, continuous, _ = get_noise(_flow_args(True), scheduler, latents)
    torch.testing.assert_close(noisy_latents.flatten() / noise.flatten(), continuous / 1000.0)
    assert continuous.dtype == torch.float32
    assert torch.any(continuous != continuous.floor())

    torch.manual_seed(123)
    _, _, legacy, _ = get_noise(_flow_args(False), scheduler, latents)
    assert legacy.dtype == torch.int64
    assert torch.all((0 <= legacy) & (legacy < 1000))


def test_cli_defaults_to_explicit_mha_and_exposes_opt_in_continuous_flow():
    for filename in ("sdxl_train.py", "train_network.py"):
        source = (ROOT / filename).read_text(encoding="utf-8-sig")
        assert '"--adapter_not_mha"' in source
        assert "adapter_already_mha" not in source
        assert 'getattr(args, "adapter_not_mha", False)' in source
        assert '"--flow_continuous_timesteps"' in source


def test_dataset_schema_exposes_artist_controls_per_subset():
    source = (ROOT / "library" / "config_util.py").read_text(encoding="utf-8-sig")
    assert "artist_balance_repeats: Union[bool, int] = False" in source
    assert "artist_tag_dropout_rate: float = 0.0" in source
    assert '"artist_balance_repeats": Any(bool, int)' in source
    assert '"artist_tag_dropout_rate": Any(float, int)' in source
