import ast
import math
from pathlib import Path
from typing import Dict, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, texts, padding=False, truncation=False, max_length=None, return_tensors=None):
        rows = [[1] + list(range(3, 3 + len(text.split()))) + [2] for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        masks = [[1] * len(row) for row in rows]
        if padding:
            width = max(len(row) for row in rows)
            for row, mask in zip(rows, masks):
                pad = width - len(row)
                row.extend([self.pad_token_id] * pad)
                mask.extend([0] * pad)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(masks)}
        return {"input_ids": rows, "attention_mask": masks}


def _source_tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _load_jina_state_test_class():
    path = ROOT / "llm_adapter_lib" / "jina" / "jina_clip_v2_states.py"
    tree = _source_tree(path)
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "JinaStates")
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"restore_text_sequence_shape", "assemble_selected_text_states"}
    ]
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"mean_pooling", "prepare_text_inputs", "inspect_text_inputs"}
    ]
    test_class = ast.ClassDef(
        name="JinaStatesUnderTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[*helpers, test_class], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"List": list, "Dict": Dict, "Tuple": Tuple, "torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["JinaStatesUnderTest"]


def _load_wrapper(filename):
    path = ROOT / filename
    tree = _source_tree(path)
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "JinaAndAdapter")
    module = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": Dict,
        "logger": _Logger(),
        "torch": torch,
        "_nonfinite_tensor_summary": lambda name, tensor: None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["JinaAndAdapter"]


def test_flash_unpadded_states_restore_and_h24_anchors_to_final_output():
    cls = _load_jina_state_test_class()
    restore = cls.mean_pooling.__globals__["restore_text_sequence_shape"]
    assemble = cls.mean_pooling.__globals__["assemble_selected_text_states"]
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    captured = [torch.full((5, 2), value) for value in (8.0, 16.0, 24.0)]
    final = torch.full((2, 4, 2), 99.0)

    restored = restore(captured[0], mask, "h8")
    selected = assemble(captured, final, mask, (8, 16, 24), 24)

    assert restored.shape == (2, 4, 2)
    assert selected.shape == (2, 3, 4, 2)
    torch.testing.assert_close(selected[:, -1], final)
    torch.testing.assert_close(restored[~mask.bool()], torch.zeros(3, 2))


def test_tokenization_truncates_then_pads_to_nearest_77():
    cls = _load_jina_state_test_class()
    jina = cls()
    jina.tokenizer = _FakeTokenizer()
    jina.max_length = 80
    prepared = jina.prepare_text_inputs([" ".join(["token"] * 100), "short"])

    assert prepared["input_ids"].shape == (2, 154)
    assert int(prepared["attention_mask"][0].sum()) == 80
    assert int(prepared["attention_mask"][1].sum()) == 3


def test_both_wrappers_dispatch_v3_selected_states_and_official_pool():
    for filename in ("sdxl_train.py", "train_network.py"):
        wrapper_cls = _load_wrapper(filename)

        class FakeJina:
            model = torch.nn.Linear(1, 1)

            def __call__(self, captions):
                return {
                    "jina_hidden_states": torch.ones(2, 77, 4),
                    "jina_hidden_states_selected_layers": torch.randn(2, 3, 77, 4),
                    "jina_mean_pooled_state": torch.zeros(2, 4),
                    "jina_final_mean_pooled_state": torch.ones(2, 4),
                    "attention_mask": torch.ones(2, 77),
                }

        class FakeV3(torch.nn.Module):
            required_hidden_state_layers = (8, 16, 24)
            hidden_state_input_key = "jina_hidden_states_selected_layers"

            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.received = None

            def forward(self, **values):
                self.received = values
                return values["jina_hidden_states_selected_layers"][:, -1], values["jina_mean_pooled_state"]

        adapter = FakeV3()
        wrapper = wrapper_cls(FakeJina(), _FakeTokenizer(), adapter)
        prompt, pooled = wrapper(["one", "two"])
        assert prompt.shape == (2, 77, 4)
        assert pooled.shape == (2, 4)
        assert "jina_hidden_states" not in adapter.received
        torch.testing.assert_close(adapter.received["jina_mean_pooled_state"], torch.zeros(2, 4))


def test_training_conditioning_reads_processed_batch_captions():
    for filename in ("sdxl_train.py", "sdxl_train_network.py"):
        source = (ROOT / filename).read_text(encoding="utf-8-sig")
        assert 'batch["captions"]' in source

