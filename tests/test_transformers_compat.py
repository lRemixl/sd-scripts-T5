import unittest
from types import SimpleNamespace

import torch
from transformers import CLIPTextConfig

from library.transformers_compat import (
    CLIPTextModel,
    TRANSFORMERS_VERSION,
    ensure_legacy_clip_symbols,
    pretrained_dtype_kwargs,
    repair_jina_clip_nonpersistent_buffers,
)


class CLIPTextModelCompatibilityTests(unittest.TestCase):
    def test_jina_nonpersistent_buffers_are_reinitialized(self):
        class FakeLora(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("lora_dropout_mask", torch.zeros(2, 3), persistent=False)

        class FakeRotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 8
                self.base = 10000.0
                self.register_buffer("inv_freq", torch.zeros(4), persistent=False)
                self._cos_cached = torch.tensor([1.0])
                self._sin_cached = torch.tensor([1.0])
                self._seq_len_cached = 10

            def _compute_inv_freq(self, device=None):
                return 1.0 / (
                    self.base
                    ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
                )

        class FakeEVARotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("freqs_cos", torch.full((9, 8), float("nan")))
                self.register_buffer("freqs_sin", torch.full((9, 8), float("nan")))

        model = torch.nn.Module()
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(pt_hw_seq_len=2),
        )
        model.lora = FakeLora()
        model.rotary = FakeRotary()
        model.vision_model = torch.nn.Module()
        model.vision_model.rope = FakeEVARotary()

        result = repair_jina_clip_nonpersistent_buffers(model)

        self.assertEqual(
            result,
            {
                "lora_dropout_masks": 1,
                "rotary_inv_freq": 1,
                "eva_rotary_freqs": 2,
                "repaired": 4,
            },
        )
        self.assertTrue(torch.equal(model.lora.lora_dropout_mask, torch.ones(2, 3)))
        self.assertTrue(torch.equal(model.rotary.inv_freq, model.rotary._compute_inv_freq()))
        self.assertTrue(torch.isfinite(model.vision_model.rope.freqs_cos).all())
        self.assertTrue(torch.isfinite(model.vision_model.rope.freqs_sin).all())
        self.assertTrue(
            torch.allclose(
                model.vision_model.rope.freqs_cos.square()
                + model.vision_model.rope.freqs_sin.square(),
                torch.ones_like(model.vision_model.rope.freqs_cos),
            )
        )
        self.assertIsNone(model.rotary._cos_cached)
        self.assertIsNone(model.rotary._sin_cached)
        self.assertEqual(model.rotary._seq_len_cached, 0)

        second_result = repair_jina_clip_nonpersistent_buffers(model)
        self.assertEqual(second_result["repaired"], 0)

    def test_pretrained_dtype_keyword_matches_transformers_major_version(self):
        kwargs = pretrained_dtype_kwargs(torch.bfloat16)

        expected_key = "dtype" if TRANSFORMERS_VERSION.major >= 5 else "torch_dtype"
        self.assertEqual(kwargs, {expected_key: torch.bfloat16})

    def test_jina_clip_loss_compatibility_symbol(self):
        ensure_legacy_clip_symbols()
        from transformers.models.clip import modeling_clip

        similarity = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        expected = torch.nn.functional.cross_entropy(similarity, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(modeling_clip.clip_loss(similarity), expected))

    def _config(self):
        return CLIPTextConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=8,
            projection_dim=8,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
        )

    def test_legacy_module_and_state_dict_layout_is_preserved(self):
        model = CLIPTextModel(self._config())

        self.assertIsNotNone(model.text_model.embeddings)
        self.assertIsNotNone(model.text_model.encoder)
        self.assertIsNotNone(model.text_model.final_layer_norm)
        self.assertTrue(any(key.startswith("text_model.embeddings.") for key in model.state_dict()))

    def test_forward_and_state_dict_round_trip(self):
        model = CLIPTextModel(self._config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 2]])

        with torch.no_grad():
            output = model(input_ids=input_ids, output_hidden_states=True)

        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 4, 8))
        self.assertIsNotNone(output.hidden_states)

        clone = CLIPTextModel(self._config())
        result = clone.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()

