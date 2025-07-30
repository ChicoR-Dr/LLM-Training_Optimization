# 🧵 TinyLlama QLoRA Training with Unsloth

This directory contains fast and memory-efficient QLoRA fine-tuning of TinyLlama using [Unsloth](https://github.com/unslothai/unsloth).

> Ideal for single-GPU training or experiments on 12–24GB VRAM GPUs.

## 📦 Requirements

```bash
pip install unsloth datasets peft transformers
```

## 🚀 Run Training

```bash

# Make sure you've installed `unsloth` and `transformers`
pip install unsloth datasets transformers

python train.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name tatsu-lab/alpaca \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --bf16 \
  --output_dir unsloth_outputs/

```

## 🧠 Dataset Format

Same Alpaca format:
```json
{
  "instruction": "Tell me a joke.",
  "input": "",
  "output": "Why did the scarecrow win an award..."
}
```

## 🧪 Run Inference

```bash
python inference.py
```

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./outputs/peft",
    load_in_4bit=True,
    max_seq_length=512,
)

FastLanguageModel.for_inference(model)

input = tokenizer("Write a haiku about mountains.", return_tensors="pt").to(model.device)
out = model.generate(**input, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## 📤 Outputs

| Folder              | Description                       |
|---------------------|-----------------------------------|
| `outputs/peft/`     | PEFT adapter weights              |
| `merged_model/`     | (Optional) merged model (if used) |

## 💡 Notes

- Ideal for Colab, RTX 4090, A100, or similar.
- For merged full model use `merge_and_save()` (see Unsloth docs).

## 🧾 License

Unsloth is Apache-2.0. This repo uses permissively licensed models and training utilities.
