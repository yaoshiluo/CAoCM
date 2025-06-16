import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
from transformers import DataCollatorForSeq2Seq


project_root = os.path.abspath(os.path.dirname(__file__))

# 训练数据 & 输出路径
data_path = os.path.join(project_root, "data/medical_o1_sft_with_prompt.jsonl")
output_dir = os.path.join(project_root, "output")

model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)



# LoRA 配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "q_proj", "v_proj"]  
)

model = get_peft_model(model, peft_config)

# 加载数据集（JSONL，每行为 {"messages": [...] }）
dataset = load_dataset("json", data_files=data_path)["train"]

def format_prompt(example):
    prompt = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        return_attention_mask=True
    )
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": tokens["input_ids"][:]  # 复制一份，后续 pad
    }


dataset = dataset.map(format_prompt)
dataset.set_format(type="torch")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
) 

# 训练参数（根据显存调节）
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    report_to="none"
)

# Trainer 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train(resume_from_checkpoint=True)


# 保存最终 LoRA 微调模型（增量）
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("LoRA 微调完成，模型保存在:", output_dir)
