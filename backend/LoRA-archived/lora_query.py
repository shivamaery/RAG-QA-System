from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
model_path = current_dir + "/LoRA-Tuned-Model"

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Set model to eval mode
model.eval()

# Prompt to query
prompt = "What is the Rayleigh-Ritz approximation technique"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
    )

output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output)