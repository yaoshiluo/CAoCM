# merge_lora_qwen.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Path settings
base_model_path = "/home/fortiss/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B-Chat/snapshots/e482ee3f73c375a627a16fdf66fd0c8279743ca6"
adapter_path = "/home/fortiss/minimind/qwen_lora_medical/output_prompt"
save_path = "/home/fortiss/minimind/qwen_lora_medical/merged_qwen"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Load LoRA adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# Save the merged model and tokenizer
print("Saving the merged model...")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved to: {save_path}")
