# inference_merged_qwen.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the merged model
model_path = "/home/fortiss/minimind/qwen_lora_medical/merged_qwen"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()

# System prompt
system_prompt = "You are a careful and accurate medical assistant who only gives evidence-based diagnoses."

# Chat loop
print("🩺 Welcome to the Medical QA Assistant (Merged Model). Type your question (Ctrl+C to exit):\n")

try:
    while True:
        user_input = input("👤 You: ").strip()
        if not user_input:
            continue

        # Construct ChatML prompt
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n🤖 Qwen: {response[len(prompt):].strip()}\n")

except KeyboardInterrupt:
    print("\n🛑 Exited the chat.")
