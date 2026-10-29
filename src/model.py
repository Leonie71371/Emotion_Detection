import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):

        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def prompt(self, messages, max_new_tokens=30):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load_model() first.")

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, do_sample = False
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response.strip()
