from mamba_ssm.models.model_wrapper import ModelWrapper
import torch
from transformers import AutoTokenizer

       
def test_model_with_custom_input(model_name, tokenizer_name, custom_prompt):
    device = "cpu"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    wrapper_model = ModelWrapper(model_name=model_name, use_generation=True, device=device, dtype=dtype)

    tokens = tokenizer(custom_prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)

    output = wrapper_model(input_ids)
    decoded_output = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    return decoded_output

if __name__ == "__main__":
    model_name = "state-spaces/mamba-130m" 
    tokenizer = "EleutherAI/gpt-neox-20b"
    custom_prompt = "Hello, world!"
    generated_text = test_model_with_custom_input(model_name, tokenizer, custom_prompt)
    print("Generated text:", generated_text)
