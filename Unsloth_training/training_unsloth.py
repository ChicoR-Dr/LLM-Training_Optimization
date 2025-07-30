from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset



model_name = '../models/TinyLlama-1.1B-Chat-v1.0/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6'


# Load base model with QLoRA enabled
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 512,
    dtype = None,         # auto-detect FP16/bfloat16
    load_in_4bit = True,  # for QLoRA
)

# Apply LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 42,
    use_rslora = False,  # You can experiment with this
    loftq_config = None,
)
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# Example dataset
def tokenize(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    tokenized_example = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized_example["labels"] = tokenized_example["input_ids"].copy()
    return tokenized_example

# Remove original columns so Trainer won’t try to convert strings to tensors
tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Set up trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=10,
        output_dir="./unsloth_outputs",
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=50,
        bf16=True,
        report_to="none",
        remove_unused_columns=False
    ),
    data_collator=data_collator,
)

# Train
trainer.train()

model.save_pretrained("./unsloth_tinyllama-qlora/peft")
tokenizer.save_pretrained("./unsloth_tinyllama-qlora/peft")

