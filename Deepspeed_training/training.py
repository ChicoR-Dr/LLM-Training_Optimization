import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import torch.distributed as dist

os.environ["NCCL_IGNORE_UNSUPPORTED"] = "1"
os.environ["DEEPSPEED_MULTINODE_LAUNCH"] = "0"

def tokenize(tokenizer, example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    tokenized_example = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized_example["labels"] = tokenized_example["input_ids"].copy()
    return tokenized_example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--deepspeed", type=str, default="config/deepspeed_config.json")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--bf16", type=bool, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_dataset(args.dataset_name, split="train[:1000]")
    tokenized = dataset.map(lambda x: tokenize(tokenizer, x))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/peft")

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning: did not destroy_process_group(): {e}")

if __name__ == "__main__":
    main()

