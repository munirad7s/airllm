import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from airllm.airllm_gemma4 import AirLLMGemma4
from airllm.utils import split_and_save_layers


class TestAirLLMGemma4(unittest.TestCase):
    def test_uses_nested_language_model_layers(self):
        model = AirLLMGemma4.__new__(AirLLMGemma4)

        model.set_layer_names_dict()

        self.assertEqual(
            model.layer_names_dict,
            {
                "embed": "model.language_model.embed_tokens",
                "layer_prefix": "model.language_model.layers",
                "norm": "model.language_model.norm",
                "lm_head": "lm_head",
            },
        )

    def test_splits_nested_language_layer_across_safetensor_shards(self):
        layer_names = {
            "embed": "model.language_model.embed_tokens",
            "layer_prefix": "model.language_model.layers",
            "norm": "model.language_model.norm",
            "lm_head": "lm_head",
        }
        first_shard = {
            "model.language_model.embed_tokens.weight": torch.arange(8).reshape(4, 2),
            "model.language_model.layers.0.left.weight": torch.ones(2, 2),
        }
        second_shard = {
            "model.language_model.layers.0.right.weight": torch.full((2, 2), 2),
            "model.language_model.norm.weight": torch.ones(2),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            first_name = "model-00001-of-00002.safetensors"
            second_name = "model-00002-of-00002.safetensors"
            save_file(first_shard, checkpoint_path / first_name)
            save_file(second_shard, checkpoint_path / second_name)
            weight_map = {
                **{key: first_name for key in first_shard},
                **{key: second_name for key in second_shard},
            }
            (checkpoint_path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8")

            split_path = split_and_save_layers(checkpoint_path, layer_names=layer_names)
            split_layer = load_file(
                str(Path(split_path) / "model.language_model.layers.0.safetensors"))

            self.assertEqual(set(split_layer), {
                "model.language_model.layers.0.left.weight",
                "model.language_model.layers.0.right.weight",
            })
            self.assertTrue(torch.equal(
                split_layer["model.language_model.layers.0.right.weight"],
                second_shard["model.language_model.layers.0.right.weight"],
            ))
