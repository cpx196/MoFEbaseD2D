from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from MoFE.upcycling import (
    UpcycledGPT2MLP,
    UpcyclingConfig,
    collect_upcycling_losses,
    convert_gpt2_to_upcycling,
    load_upcycling_checkpoint,
    save_upcycling_checkpoint,
    upcycling_parameter_breakdown,
)


def tiny_model() -> GPT2LMHeadModel:
    return GPT2LMHeadModel(
        GPT2Config(
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
    )


class UpcyclingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.dense = tiny_model().eval()
        self.model = copy.deepcopy(self.dense)
        self.config = UpcyclingConfig(
            model_name_or_path="tiny-local-gpt2",
            moe_layer_indices=(2,),
            num_experts=4,
            top_k=2,
            seed=42,
        )
        convert_gpt2_to_upcycling(self.model, self.config)
        self.model.eval()
        self.layer = self.model.transformer.h[2].mlp

    def test_experts_are_exact_independent_copies(self) -> None:
        self.assertIsInstance(self.layer, UpcycledGPT2MLP)
        dense_state = self.dense.transformer.h[2].mlp.state_dict()
        for expert in self.layer.experts:
            for name, expected in dense_state.items():
                torch.testing.assert_close(expert.state_dict()[name], expected)
        self.assertNotEqual(
            self.layer.experts[0].c_fc.weight.data_ptr(),
            self.layer.experts[1].c_fc.weight.data_ptr(),
        )

    def test_conversion_preserves_eval_logits(self) -> None:
        input_ids = torch.randint(0, 64, (2, 8))
        with torch.inference_mode():
            expected = self.dense(input_ids).logits
            actual = self.model(input_ids).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_topk_weights_and_gradients(self) -> None:
        self.model.train()
        input_ids = torch.randint(0, 64, (2, 8))
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        state = self.layer.routing_state
        self.assertIsNotNone(state)
        self.assertEqual(tuple(state.topk_indices.shape), (16, 2))
        self.assertEqual(int(state.assignment_counts.sum()), 32)
        aux = collect_upcycling_losses(self.model)
        loss = outputs.loss + 0.01 * aux["balance_loss"] + 0.001 * aux["z_loss"]
        loss.backward()
        self.assertIsNotNone(self.layer.router.weight.grad)
        self.assertTrue(torch.isfinite(self.layer.router.weight.grad).all())
        expert_gradients = [
            expert.c_fc.weight.grad for expert in self.layer.experts
        ]
        self.assertTrue(any(gradient is not None for gradient in expert_gradients))

    def test_parameter_breakdown_and_checkpoint_round_trip(self) -> None:
        breakdown = upcycling_parameter_breakdown(self.model)
        self.assertEqual(breakdown["total"], sum(p.numel() for p in self.model.parameters()))
        input_ids = torch.randint(0, 64, (1, 8))
        with torch.inference_mode():
            expected = self.model(input_ids).logits
        with tempfile.TemporaryDirectory() as temporary_directory:
            save_upcycling_checkpoint(
                self.model, Path(temporary_directory), self.config
            )
            restored, restored_config = load_upcycling_checkpoint(temporary_directory)
            restored.eval()
            with torch.inference_mode():
                actual = restored(input_ids).logits
        self.assertEqual(restored_config, self.config)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
