import json
from pathlib import Path

import pytest

from library import transformers_compat


ROOT = Path(__file__).resolve().parents[1]


class _PreTokenizer:
    def __init__(self, component_types):
        self.component_types = component_types

    def __getstate__(self):
        return json.dumps(
            {
                "type": "Sequence",
                "pretokenizers": [{"type": component_type} for component_type in self.component_types],
            }
        ).encode("utf-8")


class _Backend:
    def __init__(self, component_types):
        self.pre_tokenizer = _PreTokenizer(component_types)


class _Tokenizer:
    def __init__(self, component_types=("WhitespaceSplit", "Metaspace")):
        self.backend_tokenizer = _Backend(component_types)


def test_jina_tokenizer_loader_enforces_official_xlmr_contract(monkeypatch):
    calls = []

    def from_pretrained(model_name_or_path, **kwargs):
        calls.append((model_name_or_path, kwargs))
        return _Tokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", from_pretrained)
    tokenizer = transformers_compat.load_jina_clip_tokenizer(
        "jinaai/jina-clip-v2",
        local_files_only=True,
        revision="model-commit",
    )

    assert isinstance(tokenizer, _Tokenizer)
    assert calls == [
        (
            "jinaai/jina-clip-v2",
            {
                "trust_remote_code": True,
                "fix_mistral_regex": False,
                "local_files_only": True,
                "revision": "model-commit",
            },
        )
    ]


def test_jina_tokenizer_contract_rejects_mistral_split_patch():
    with pytest.raises(RuntimeError, match="Mistral regex repair"):
        transformers_compat.validate_jina_clip_tokenizer(_Tokenizer(("Split", "Metaspace")))

    with pytest.raises(ValueError, match="must not use"):
        transformers_compat.load_jina_clip_tokenizer("jinaai/jina-clip-v2", fix_mistral_regex=True)


def test_jina_load_context_constrains_nested_trusted_code(monkeypatch):
    from transformers import AutoModel, AutoTokenizer
    from transformers.models.auto import auto_factory

    tokenizer_calls = []
    dynamic_calls = []
    model_config_calls = []

    def tokenizer_loader(cls, model_name_or_path, **kwargs):
        tokenizer_calls.append((model_name_or_path, kwargs))
        return _Tokenizer()

    def dynamic_loader(*args, **kwargs):
        dynamic_calls.append((args, kwargs))
        return object

    def model_from_config(cls, config, **kwargs):
        model_config_calls.append((config, kwargs))
        return object()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(tokenizer_loader))
    monkeypatch.setattr(AutoModel, "from_config", classmethod(model_from_config))
    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", dynamic_loader)

    with transformers_compat.jina_clip_load_context(local_files_only=True):
        AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        AutoModel.from_config(object(), code_revision=None, marker="none")
        AutoModel.from_config(object(), code_revision="code-commit", marker="pinned")
        auto_factory.get_class_from_dynamic_module(
            "modeling_lora.XLMRobertaLoRA",
            "jinaai/xlm-roberta-flash-implementation",
        )

    assert tokenizer_calls == [
        (
            "jinaai/jina-embeddings-v3",
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "fix_mistral_regex": False,
            },
        )
    ]
    assert model_config_calls[0][1] == {"marker": "none"}
    assert model_config_calls[1][1] == {
        "code_revision": "code-commit",
        "marker": "pinned",
    }
    assert dynamic_calls[0][1]["local_files_only"] is True


def test_jina_state_module_has_no_import_time_network_or_offline_override():
    source = (ROOT / "llm_adapter_lib" / "jina" / "jina_clip_v2_states.py").read_text(encoding="utf-8")
    assert "requests.get(" not in source
    assert "TRANSFORMERS_OFFLINE" not in source
    assert "load_jina_clip_tokenizer(" in source
    assert "jina_clip_load_context(" in source
    assert "Path(model_id).is_dir()" in source
    assert '"trust_remote_code": JINA_CLIP_TRUST_REMOTE_CODE' in source


def test_training_entrypoints_expose_revision_pins_and_jina_loader():
    for filename in ("sdxl_train.py", "train_network.py"):
        source = (ROOT / filename).read_text(encoding="utf-8-sig")
        assert '"--llm_model_revision"' in source
        assert '"--llm_code_revision"' in source
        assert "load_jina_clip_tokenizer(" in source
        assert 'revision=getattr(args, "llm_model_revision", None)' in source
        assert 'code_revision=getattr(args, "llm_code_revision", None)' in source

