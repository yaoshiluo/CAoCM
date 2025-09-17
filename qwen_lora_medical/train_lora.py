import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
from transformers import DataCollatorForSeq2Seq


project_root = os.path.abspath(os.path.dirname(__file__))

# Training data & output path
data_path = os.path.join(project_root, "data/medical_o1_sft_with_prompt.jsonl")
output_dir = os.path.join(project_root, "output_prompt")

model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "q_proj", "v_proj"]  
)

model = get_peft_model(model, peft_config)

# Load dataset (JSONL, each line is {"messages": [...] })
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
        "labels": tokens["input_ids"][:]  # copy for padding later
    }


dataset = dataset.map(format_prompt)
dataset.set_format(type="torch")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
) 

# Training arguments (adjust according to GPU memory)
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

# Start training with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train(resume_from_checkpoint=True)


# Save the final LoRA fine-tuned model (delta weights)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("LoRA fine-tuning finished. Model saved to:", output_dir)
