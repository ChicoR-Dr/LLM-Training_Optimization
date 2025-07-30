# 🚀 LLM Training Optimization Projects

This repository demonstrates two efficient methods for fine-tuning Large Language Models (LLMs) using:

- ⚡️ **DeepSpeed** with Transformers and PEFT (QLoRA)
- 🐍 **Unsloth**: An ultra-fast LoRA/QLoRA trainer

Each method is encapsulated in its own directory with training scripts and configs.

---

## 📁 Project Structure

```
LLM-Training_Optimization/
│
├── deepspeed/               # Fine-tuning with DeepSpeed + QLoRA
│   ├── train.py
│   ├── config/
│   │   └── deepspeed_config.json
│   └── README.md
│
├── unsloth/                 # Fine-tuning with Unsloth + QLoRA
│   ├── train.py
│   └── README.md
│
└── README.md                # This file
```

---

## ✅ Dataset

Both projects use the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) for demonstration. You can change the dataset in `train.py`.

---

## 🧠 Base Model

These examples use [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), a 1.1B parameter instruction-tuned model, suitable for low-resource fine-tuning.

---

## 🛠 Requirements

Install the required libraries in a fresh environment:

```bash
pip install -r requirements.txt
```

Here's a minimal example `requirements.txt`:

```text
transformers
datasets
peft
accelerate
deepspeed
unsloth
```

---

## 📦 How to Use

### ➤ DeepSpeed Training

```bash
cd deepspeed
deepspeed train.py \
  --deepspeed config/deepspeed_config.json \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name tatsu-lab/alpaca \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --bf16 True \
  --output_dir outputs/
```

> See [`deepspeed/README.md`](./deepspeed/README.md) for full instructions.

---

### ➤ Unsloth Training

```bash
cd unsloth
python train.py
```

> See [`unsloth/README.md`](./unsloth/README.md) for full instructions.

---

## 📤 Output

Each script saves the fine-tuned model in:

- `./deepspeed/outputs/`
- `./unsloth/unsloth_tinyllama-qlora/peft/`

These can be pushed to 🤗 Hub or used for inference.

---

## 📘 Notes

- Both use **QLoRA** (4-bit quantization + LoRA).
- Suitable for training on a **single GPU with ~16 GB VRAM**.
- You can extend either script for more epochs, larger datasets, or custom prompts.

---

## 👨‍💻 Author

Chinmay @ Cognisyn Labs  
🔬 https://cognisynlabs.com/

---

## 🧪 License

MIT License
