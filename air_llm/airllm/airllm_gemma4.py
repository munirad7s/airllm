from .airllm_base import AirLLMBaseModel


class AirLLMGemma4(AirLLMBaseModel):
    """Stream the language model inside a Gemma 4 multimodal checkpoint."""

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": "model.language_model.embed_tokens",
            "layer_prefix": "model.language_model.layers",
            "norm": "model.language_model.norm",
            "lm_head": "lm_head",
        }
