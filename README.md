# LLM-Training_Optimization


.
├── deepspeed/
│   ├── train.py
│   ├── config/
│   │   └── deepspeed_config.json
│   ├── outputs/
│   └── README.md  <-- For DeepSpeed
├── unsloth/
│   ├── train.py
│   ├── inference.py
│   ├── outputs/
│   └── README.md  <-- For Unsloth
├── shared/
│   └── utils.py
└── .gitignore



# 🦙 TinyLlama QLoRA Fine-Tuning with DeepSpeed

This directory demonstrates how to fine-tune [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using QLoRA and DeepSpeed. It supports multi-GPU training and full model fine-tuning.

---

## 📦 Requirements

```bash
pip install transformers peft accelerate deepspeed datasets bitsandbytes


⚙️ Configuration
Edit config/deepspeed_config.json to control:

ZeRO stage

Offloading

Gradient checkpointing

Memory optimizations

Model details
Tinylamma 1b already donwloaded on local system

🚀 Launch Training

1 python train.py

2 deepspeed train.py \
  --deepspeed config/deepspeed_config.json \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name tatsu-lab/alpaca \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --bf16 True \
  --output_dir outputs/
You can toggle between full fine-tuning and LoRA by modifying the trainer logic inside train.py.


🧠 Dataset Format
Uses Alpaca-style instruction tuning format:

{
  "instruction": "Describe photosynthesis.",
  "input": "",
  "output": "Photosynthesis is the process..."
}

📤 Outputs
Folder	Description
outputs/	Hugging Face checkpoints (PEFT adapters)
merged_model/	Optional merged full model post-training
