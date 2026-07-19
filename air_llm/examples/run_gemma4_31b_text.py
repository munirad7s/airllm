import os

import torch

from airllm import AutoModel


MODEL_ID = "google/gemma-4-31b-it"

model = AutoModel.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    hf_token=os.getenv("HF_TOKEN"),
)

messages = [{"role": "user", "content": "Antworte nur mit dem Wort: Bereit"}]
prompt = model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = model.tokenizer(prompt, return_tensors="pt")
inputs = {name: tensor.to(model.running_device) for name, tensor in inputs.items()}

output = model.generate(
    **inputs,
    max_new_tokens=2,
    do_sample=False,
    use_cache=True,
)
print(model.tokenizer.decode(output[0], skip_special_tokens=True))
