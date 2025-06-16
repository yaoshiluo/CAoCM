from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Paths to the base model and LoRA adapter
base_model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B-Chat/snapshots/e482ee3f73c375a627a16fdf66fd0c8279743ca6"
adapter_path = "output"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model and apply LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path).eval()

# Interactive chat
print("🩺 Welcome to the Medical QA Assistant (LoRA fine-tuned). Type your question below (Ctrl+C to exit):\n")
system_prompt = "You are a careful and accurate medical assistant who only gives evidence-based diagnoses."

try:
    while True:
        user_input = input("👤 You: ").strip()
        if not user_input:
            continue

        # Build ChatML format prompt
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

        # Tokenize and generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n🤖 Qwen: {response[len(prompt):].strip()}\n")

except KeyboardInterrupt:
    print("\n🛑 Exiting the chat.")
