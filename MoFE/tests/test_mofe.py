from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from MoFE.checkpoint import load_mofe_checkpoint, save_mofe_checkpoint
from MoFE.config import MoFEConfig
from MoFE.layer import MoFEGPT2MLP
from MoFE.modeling import collect_mofe_losses, convert_gpt2_to_mofe


def tiny_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=3,
        n_head=4,
        n_inner=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=1,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(config)


def tiny_mofe_config() -> MoFEConfig:
    return MoFEConfig(
        model_name_or_path="tiny-local-gpt2",
        moe_layer_indices=(2,),
        num_private_experts=16,
        top_k=3,
        rank=12,
        core_init_std=0.025,
        seed=42,
    )


class MoFETest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.dense = tiny_model().eval()
        self.original_mlp = copy.deepcopy(self.dense.transformer.h[2].mlp)
        self.config = tiny_mofe_config()
        self.model = copy.deepcopy(self.dense)
        convert_gpt2_to_mofe(self.model, self.config)
        self.model.eval()
        self.layer = self.model.transformer.h[2].mlp

    def test_replacement_and_dense_initialization(self) -> None:
        self.assertIsInstance(self.layer, MoFEGPT2MLP)
        for name, expected in self.original_mlp.state_dict().items():
            torch.testing.assert_close(self.layer.shared_expert.state_dict()[name], expected)

        dense_w1 = self.original_mlp.c_fc.weight.T
        dense_w2 = self.original_mlp.c_proj.weight.T
        for group in range(4):
            torch.testing.assert_close(self.layer.a1[group], dense_w1[:, :12])
            torch.testing.assert_close(self.layer.b1[group], dense_w1[:12, :])
            torch.testing.assert_close(self.layer.a2[group], dense_w2[:, :12])
            torch.testing.assert_close(self.layer.b2[group], dense_w2[:12, :])
        self.assertGreater(self.layer.core1.std().item(), 0.0)
        self.assertFalse(torch.equal(self.layer.core1[0], self.layer.core1[1]))
        self.assertEqual(torch.count_nonzero(self.layer.core2).item(), 0)

    def test_factorized_and_materialized_expert_match(self) -> None:
        states = torch.randn(5, 16)
        expert_index = 7
        factorized = self.layer.private_expert_forward(states, expert_index)
        weight1, weight2 = self.layer.materialize_expert_weights(expert_index)
        materialized = F.linear(states, weight1, self.layer.private_bias1[expert_index])
        materialized = self.layer.shared_expert.act(materialized)
        materialized = F.linear(
            materialized, weight2, self.layer.private_bias2[expert_index]
        )
        torch.testing.assert_close(factorized, materialized, atol=1e-6, rtol=1e-5)

    def test_zero_output_core_preserves_dense_logits(self) -> None:
        input_ids = torch.randint(0, 64, (2, 8))
        with torch.inference_mode():
            expected = self.dense(input_ids).logits
            actual = self.model(input_ids).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_token_choice_routing_and_gradients(self) -> None:
        self.model.train()
        input_ids = torch.randint(0, 64, (2, 8))
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        state = self.layer.routing_state
        self.assertIsNotNone(state)
        self.assertEqual(tuple(state.topk_indices.shape), (16, 3))
        self.assertEqual(int(state.assignment_counts.sum()), 16 * 3)
        self.assertTrue(torch.isfinite(state.private_to_shared_norm))

        aux = collect_mofe_losses(self.model)
        loss = outputs.loss + 0.01 * aux["balance_loss"] + 0.001 * aux["z_loss"]
        loss.backward()
        self.assertGreater(self.layer.core2.grad.abs().sum().item(), 0.0)
        for parameter in (
            self.layer.a1,
            self.layer.b1,
            self.layer.core1,
            self.layer.a2,
            self.layer.b2,
            self.layer.core2,
            self.layer.private_bias1,
            self.layer.private_bias2,
            self.layer.router.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_checkpoint_round_trip(self) -> None:
        input_ids = torch.randint(0, 64, (1, 8))
        with torch.inference_mode():
            expected = self.model(input_ids).logits
        with tempfile.TemporaryDirectory() as temporary_directory:
            save_mofe_checkpoint(
                self.model, Path(temporary_directory), self.config
            )
            restored, restored_config = load_mofe_checkpoint(temporary_directory)
            restored.eval()
            with torch.inference_mode():
                actual = restored(input_ids).logits
        self.assertEqual(restored_config, self.config)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
