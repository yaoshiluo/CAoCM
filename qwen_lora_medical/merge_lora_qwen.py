# merge_lora_qwen.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 路径设置
base_model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B-Chat/snapshots/e482ee3f73c375a627a16fdf66fd0c8279743ca6"
adapter_path = "output"
save_path = "merged_qwen"

# 加载 tokenizer 和基础模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# 加载 LoRA adapter 并合并
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# 保存合并后的模型和 tokenizer
print("💾 正在保存合并后的模型...")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 已保存到：{save_path}")
