import unittest
from unittest import mock

import torch

from llm_adapter_lib.jina.jina_to_sdxl_adapter_v2 import (
    ExplicitMultiheadAttention,
    JinaToSDXLAdapterV2,
)
from llm_adapter_lib.jina.jina_to_sdxl_adapter_v3 import (
    JinaToSDXLAdapterV3,
    LayerwiseAttentionFusion,
    filter_compatible_adapter_state_dict,
    missing_or_mismatched_v3_keys,
)


def _adapter(n_attention_blocks=1):
    return JinaToSDXLAdapterV3(
        llm_dim=4,
        sdxl_seq_dim=8,
        sdxl_pooled_dim=6,
        n_attention_blocks=n_attention_blocks,
        num_heads=2,
        dropout=0,
        max_seq_len=5,
    )


class LayerwiseAttentionFusionTests(unittest.TestCase):
    def test_large_token_batch_avoids_sdpa_grid_limit(self):
        attention = ExplicitMultiheadAttention(embed_dim=2, num_heads=2, dropout=0)
        # 32,768 * 2 heads is the first value above the CUDA grid limit used
        # by the fallback.  V3 reaches this shape after folding B*tokens into
        # the layer-fusion batch dimension.
        states = torch.randn(32768, 3, 2, requires_grad=True)

        with mock.patch(
            "llm_adapter_lib.jina.jina_to_sdxl_adapter_v2.F.scaled_dot_product_attention",
            side_effect=AssertionError("large folded batches must use eager attention"),
        ):
            output, weights = attention(states, states, states)
            output.square().mean().backward()

        self.assertIsNone(weights)
        self.assertEqual(tuple(output.shape), tuple(states.shape))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(states.grad).all())

    def test_pooling_is_token_local_and_weights_sum_to_one(self):
        fusion = LayerwiseAttentionFusion(
            dim=4,
            num_layers=3,
            num_blocks=2,
            num_heads=2,
            dropout=0,
        ).eval()
        states = torch.randn(2, 3, 5, 4)
        delta, weights = fusion(states)

        self.assertEqual(tuple(delta.shape), (2, 5, 4))
        self.assertEqual(tuple(weights.shape), (2, 5, 3))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))
        self.assertTrue(torch.all(weights > 0))

    def test_perturbing_one_token_does_not_change_other_token_fusion(self):
        fusion = LayerwiseAttentionFusion(dim=4, num_layers=3, num_blocks=2, num_heads=2).eval()
        first = torch.randn(1, 3, 4, 4)
        second = first.clone()
        second[:, :, 3] += 1000

        delta_first, _ = fusion(first)
        delta_second, _ = fusion(second)
        torch.testing.assert_close(delta_first[:, :3], delta_second[:, :3])


class JinaToSDXLAdapterV3Tests(unittest.TestCase):
    def test_zero_gate_uses_exact_h24_sequence(self):
        adapter = _adapter(n_attention_blocks=0).eval()
        selected = torch.randn(2, 3, 5, 4)
        pooled = torch.randn(2, 4)
        mask = torch.ones(2, 5, dtype=torch.long)

        prompt, _ = adapter(selected, pooled, mask)
        expected = adapter.seq_projection(selected[:, -1])
        torch.testing.assert_close(prompt, expected)
        torch.testing.assert_close(adapter.layer_fusion.channel_gate, torch.zeros(4))

    def test_v2_checkpoint_is_an_exact_valid_token_sequence_warm_start(self):
        torch.manual_seed(7)
        v2 = JinaToSDXLAdapterV2(
            llm_dim=4,
            sdxl_seq_dim=8,
            sdxl_pooled_dim=6,
            n_attention_blocks=1,
            num_heads=2,
            dropout=0,
            max_seq_len=5,
        ).eval()
        v3 = _adapter().eval()
        compatible, unexpected, shape_mismatches = filter_compatible_adapter_state_dict(v3, v2.state_dict())
        v3.load_state_dict(compatible, strict=False)

        h24 = torch.randn(2, 5, 4)
        selected = torch.stack([torch.randn_like(h24), torch.randn_like(h24), h24], dim=1)
        pooled = torch.randn(2, 4)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        with torch.no_grad():
            v2_prompt, _ = v2(h24, pooled, mask)
            v3_prompt, _ = v3(selected, pooled, mask)

        valid = mask.bool().unsqueeze(-1).expand_as(v2_prompt)
        torch.testing.assert_close(v3_prompt[valid], v2_prompt[valid])
        self.assertTrue(any(key.startswith("attention_pooler.") for key in unexpected))
        self.assertEqual(shape_mismatches, [])

    def test_mask_blocks_padding_influence_and_zeroes_padded_queries(self):
        adapter = _adapter().eval()
        selected = torch.randn(2, 3, 5, 4)
        perturbed = selected.clone()
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        padding = ~mask.bool().unsqueeze(1).unsqueeze(-1).expand_as(perturbed)
        perturbed[padding] = torch.randn_like(perturbed[padding]) * 10000
        pooled = torch.randn(2, 4)

        with torch.no_grad():
            first, _ = adapter(selected, pooled, mask)
            second, _ = adapter(perturbed, pooled, mask)

        valid = mask.bool().unsqueeze(-1).expand_as(first)
        padded = ~valid
        torch.testing.assert_close(first[valid], second[valid])
        torch.testing.assert_close(first[padded], torch.zeros_like(first[padded]))
        torch.testing.assert_close(second[padded], torch.zeros_like(second[padded]))

    def test_all_padding_is_finite_and_zero(self):
        adapter = _adapter().eval()
        prompt, pooled = adapter(
            torch.randn(2, 3, 5, 4),
            torch.randn(2, 4),
            torch.zeros(2, 5),
        )
        self.assertTrue(torch.isfinite(prompt).all())
        self.assertTrue(torch.isfinite(pooled).all())
        torch.testing.assert_close(prompt, torch.zeros_like(prompt))

    def test_unmasked_operation_is_supported(self):
        prompt, pooled = _adapter().eval()(torch.randn(2, 3, 5, 4), torch.randn(2, 4))
        self.assertEqual(tuple(prompt.shape), (2, 5, 8))
        self.assertEqual(tuple(pooled.shape), (2, 6))

    def test_pooled_branch_uses_only_official_pooled_input(self):
        adapter = _adapter().eval()
        selected_a = torch.randn(2, 3, 5, 4)
        selected_b = selected_a + 100
        pooled_input = torch.randn(2, 4)
        _, pooled_a = adapter(selected_a, pooled_input)
        _, pooled_b = adapter(selected_b, pooled_input)
        torch.testing.assert_close(pooled_a, pooled_b)

    def test_gate_then_fusion_stack_receive_gradients(self):
        torch.manual_seed(11)
        adapter = _adapter(n_attention_blocks=0)
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        selected = torch.randn(2, 3, 5, 4)
        pooled = torch.randn(2, 4)

        prompt, pooled_output = adapter(selected, pooled)
        (prompt.square().mean() + pooled_output.square().mean()).backward()
        self.assertGreater(float(adapter.layer_fusion.channel_gate.grad.abs().sum()), 0)
        self.assertEqual(float(adapter.layer_fusion.layer_score.weight.grad.abs().sum()), 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        prompt, _ = adapter(selected, pooled)
        prompt.square().mean().backward()
        self.assertGreater(float(adapter.layer_fusion.layer_score.weight.grad.abs().sum()), 0)

    def test_shape_validation(self):
        adapter = _adapter()
        with self.assertRaisesRegex(ValueError, "h8/h16/h24"):
            adapter(torch.randn(2, 4, 5, 4), torch.randn(2, 4))
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            adapter(torch.randn(2, 3, 5, 4), torch.randn(2, 4), torch.ones(2, 4))

    def test_complete_checkpoint_detection(self):
        adapter = _adapter()
        state = adapter.state_dict()
        self.assertEqual(missing_or_mismatched_v3_keys(adapter, state), [])
        incomplete = dict(state)
        incomplete.pop("layer_fusion.channel_gate")
        self.assertIn("layer_fusion.channel_gate", missing_or_mismatched_v3_keys(adapter, incomplete))


if __name__ == "__main__":
    unittest.main()

