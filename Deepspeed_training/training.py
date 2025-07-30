from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os
import torch.distributed as dist

#because i only have one gpu. deep speed is more for multi gpu until explicitly mentioned
os.environ["NCCL_IGNORE_UNSUPPORTED"] = "1"
os.environ["DEEPSPEED_MULTINODE_LAUNCH"] = "0"

model_name = '../models/TinyLlama-1.1B-Chat-v1.0/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6'
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")  # small test slice

# Preprocess
'''
def tokenize(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
'''

def tokenize(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    tokenized_example = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    # Add labels identical to input_ids for causal LM
    tokenized_example["labels"] = tokenized_example["input_ids"].copy()
    return tokenized_example



tokenized = dataset.map(tokenize)

# Model: 4-bit quant + LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Trainer
args = TrainingArguments(
    output_dir="./tinyllama-qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    deepspeed="ds_config_zero2.json",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized
)

trainer.train()
model.save_pretrained("./tinyllama-qlora/peft")


# to avoid leaking resources. destroy_process_group() was not called before program exit

try:
    if dist.is_initialized():
        dist.destroy_process_group()
except Exception as e:
    print(f"Warning did not destroy_process_group(): {e}")

