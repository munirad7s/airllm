import unittest
from types import SimpleNamespace
from unittest.mock import patch

from airllm.auto_model import AutoModel



class TestAutoModel(unittest.TestCase):
    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            "ChatGLMForConditionalGeneration": "AirLLMChatGLM",
            "QWenLMHeadModel": "AirLLMQWen",
            "InternLMForCausalLM": "AirLLMInternLM",
            "BaichuanForCausalLM": "AirLLMBaichuan",
            "Gemma4ForConditionalGeneration": "AirLLMGemma4",
            "MistralForCausalLM": "AirLLMBaseModel",
        }

        for architecture, expected_class in mapping_dict.items():
            with self.subTest(architecture=architecture), patch(
                "airllm.auto_model.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(architectures=[architecture]),
            ):
                module, model_class = AutoModel.get_module_class("test/model")
                self.assertEqual(module, "airllm")
                self.assertEqual(model_class, expected_class)

