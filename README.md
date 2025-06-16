# Medical QA with Qwen1.5 + LoRA Fine-tuning

This project demonstrates how to fine-tune the Qwen1.5-1.8B-Chat model using LoRA on a medical QA dataset. It includes data preparation, training, inference via command line, and visualization with a web interface.

---

## Project Structure

```
CAoCM/
├── minimind_env.yaml                   # Conda environment definition
├── qwen_lora_medical/                  # Main project code
│   ├── convert_to_chatml.py            # Script to convert raw dataset to ChatML format
│   ├── data/
│   │   ├── medical_o1_sft.json         # Raw dataset from Hugging Face
│   │   └── medical_o1_sft_with_prompt.jsonl  # Processed dataset with prompt
│   ├── inference_lora_qwen.py          # Inference script for LoRA-only model
│   ├── train_lora.py                   # LoRA fine-tuning script
│   └── web_demo.py                     # Streamlit web demo using merged model
├── README.md
```

---

## Step-by-Step Guide

### 1. Create and activate Conda environment

```bash
conda env create -f minimind_env.yaml
conda activate minimind
```

---

### 2. Prepare the dataset

Raw data is downloaded from [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT):

```
qwen_lora_medical/data/medical_o1_sft.json
```

Convert to ChatML format required for training:

```bash
cd qwen_lora_medical
python convert_to_chatml.py
```

This will generate:

```
qwen_lora_medical/data/medical_o1_sft_with_prompt.jsonl
```

---

### 3. Train the model with LoRA

In `train_lora.py`, training data path and output directory are configured:

```python
project_root = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(project_root, "data/medical_o1_sft_with_prompt.jsonl")
output_dir = os.path.join(project_root, "output")
```

The script will automatically download the base model `Qwen/Qwen1.5-1.8B-Chat` from Hugging Face. Run training:

```bash
python train_lora.py
```

---

### 4. Merge LoRA adapter with base model

After training, merge the fine-tuned adapter with the base model:

```bash
python merge_lora_qwen.py
```

The merged model will be saved in:

```
qwen_lora_medical/merged_qwen
```

---

### 5. Inference via CLI (LoRA adapter only)

```bash
python inference_lora_qwen.py
```

This runs inference using the base model + LoRA adapter (`output_prompt` folder).

---

### 6. Web Demo via Streamlit (merged model)

Launch web interface using the merged model:

```bash
streamlit run web_demo.py --server.address 0.0.0.0 --server.port 8502
```

---

## Model Files

- `output/`: Contains LoRA fine-tuned adapter.
- `merged_qwen/`: Contains the merged full model (base + LoRA), used for web demo.

> ⚠️ Note: `inference_lora_qwen.py` uses the **unmerged model**, while `web_demo.py` uses the **merged model**.

---

## 🔗 Dataset Reference

- [FreedomIntelligence/medical-o1-reasoning-SFT on Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)

---

## 🔗 Source Code Reference

- [jingyaogong/minimind on GitHub](https://github.com/jingyaogong/minimind)


