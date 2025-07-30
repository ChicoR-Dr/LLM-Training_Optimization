# 🦙 TinyLlama QLoRA Fine-Tuning with DeepSpeed

This directory demonstrates how to fine-tune [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using QLoRA and DeepSpeed. It supports multi-GPU training and full model fine-tuning.

## 📦 Requirements

```bash
pip install transformers peft accelerate deepspeed datasets bitsandbytes
```

## ⚙️ Configuration

Edit `config/deepspeed_config.json` to control:
- ZeRO stage
- Offloading
- Gradient checkpointing
- Memory optimizations

## 🚀 Launch Training

```bash

# Single GPU basic training (just runs the script)
python train.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name tatsu-lab/alpaca \
  --output_dir outputs/

# Multi-GPU / with DeepSpeed
deepspeed train.py \
  --deepspeed config/deepspeed_config.json \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name tatsu-lab/alpaca \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --bf16 True \
  --output_dir outputs/

```



## 🧠 Dataset Format

Uses Alpaca-style instruction tuning format:
```json
{
  "instruction": "Describe photosynthesis.",
  "input": "",
  "output": "Photosynthesis is the process..."
}
```

## 📤 Outputs

| Folder              | Description                              |
|---------------------|------------------------------------------|
| `outputs/`          | Hugging Face checkpoints (PEFT adapters) |
| `merged_model/`     | Optional merged full model post-training |

## 🧠 Inference (Shared)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base, "outputs/checkpoint-100")

model.eval()
input = tokenizer("What is photosynthesis?", return_tensors="pt").to(model.device)
out = model.generate(**input, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## 🧾 License

Apache-2.0 or MIT depending on tools used (see individual packages).
