# unsloth_tinyllama/train.py
import argparse
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth TinyLlama QLoRA training")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--output_dir", type=str, default="./unsloth_outputs")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Load and tokenize dataset
    dataset = load_dataset(args.dataset_name, split="train[:1000]")

    def tokenize(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        tokenized_example = tokenizer(prompt, truncation=True, padding="max_length", max_length=args.max_seq_length)
        tokenized_example["labels"] = tokenized_example["input_ids"].copy()
        return tokenized_example

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=args.save_steps,
            bf16=args.bf16,
            report_to="none",
            remove_unused_columns=False
        ),
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(f"{args.output_dir}/peft")
    tokenizer.save_pretrained(f"{args.output_dir}/peft")


if __name__ == "__main__":
    main()

