"""
Training_unsloth.py — Modular QLoRA training script using Unsloth + TRL (SFTTrainer)
Author: Chinmay Rane
Date: 2025

Usage:
    python Training_unsloth.py --config car_config.json
"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
#from transformers import Trainer, DataCollatorForLanguageModeling #using trl -> SFTTrainer
from datasets import Dataset
from typing import Dict, Any
import random

class QLoRATrainer:
    """
    A reusable class for fine-tuning 4-bit quantized LLMs with QLoRA using Unsloth.

    It encapsulates:
    - Loading and preparing a base model
    - Attaching LoRA adapters
    - Loading JSON datasets
    - Formatting prompts
    - Training with TRL’s SFTTrainer
    - Saving LoRA adapter weights

    Config file for each of the training is needed. KIndly refer the Training_Config_Car.json in the folder for example
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize model paths, configuration, and load base model."""

        # ---- Core Config ----
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Input / Output Paths
        self.model_name = config["model_name"]
        self.data_path = Path(config["data_path"])
        self.save_path = Path(config["save_path"])
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Model setup
        self.max_seq_length = config.get("max_seq_length", 2048)
        self.dtype = None
        self.load_in_4bit = True

        # ---- Load Base Model ----
        print(f"🚀 Loading base model: {self.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )

    # --------------------------------------------------------------------
    def prepare_lora(self):
        """
        Attach LoRA adapters to the base model for efficient fine-tuning.

       
        """
        print("🔧 Attaching LoRA adapters for QLoRA training...")

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.cfg.get("lora_r", 64),  # Low-rank dimension
            target_modules=self.cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
            ),
            lora_alpha=self.cfg.get("lora_alpha", 16),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias="none",
            use_gradient_checkpointing="unsloth",  # Memory-efficient checkpointing
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        print("✅ LoRA adapters attached successfully.")

    # --------------------------------------------------------
    # Load JSON datasets
    # --------------------------------------------------------
    def load_datasets(self):
        """
        Load single JSON file with qna structure and split into train/val.
        Expected format: {"qna": [{"question": "...", "answer": "..."}]} with tr except feature for any failures of data
        """
        print("📚 Loading dataset...")
        
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as e:
            print(f"❌ Error: Dataset file not found - {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format - {e}")
            raise
        
        # Extract the qna list from the JSON
        qna_data = data.get("qna", [])
        
        if not qna_data:
            raise ValueError("❌ Error: 'qna' key not found or empty in dataset") # important as the current structure has 'qna'
        
        print(f"📊 Found {len(qna_data)} total samples")
        
        # Process and clean data
        training_data = []
        skipped = 0
        
        for i, Q_A_data in enumerate(qna_data):
            try:
                ques = Q_A_data.get("question", "").strip()
                ans = Q_A_data.get("answer", "").strip()
                
                
                
                training_data.append({
                    "question": ques,
                    "answer": ans
                })
                # Skip empty questions or answers. Originally this should be an error
                if not ques or not ans:
                    print(f"Skipping empty entry at index {i}")
                    skipped += 1
                    continue
            except Exception as e:
                print(f" Error at index {i}: {str(e)}")
                skipped += 1
                continue
        
        print(f"✅ Processed {len(training_data)} samples ({skipped} skipped)")
        
        # Random shuffle
        random.seed(self.cfg.get("random_seed", 3407)) # if need add this to config file but curretnly default is 3407
        random.shuffle(training_data)
        
        # Split into train and val
        split_ratio = self.cfg.get("train_split", 0.9)  # # if need add this to config file but curretnly default is 90% train, 10% val
        num_train = int(len(training_data) * split_ratio)
        
        train_data = training_data[:num_train]
        val_data = training_data[num_train:]
        
        print(f"📊 Split: {len(train_data)} training, {len(val_data)} validation")
        
        # Optional: Save split datasets for reference
        if self.cfg.get("save_split_files", False): # if need add this to config file but curretnly default is False
            train_file = self.save_path / "train_split.json"
            val_file = self.save_path / "val_split.json"
            
            with open(train_file, "w", encoding="utf-8") as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            with open(val_file, "w", encoding="utf-8") as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved splits to {train_file} and {val_file}")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        return train_dataset, val_dataset

    def format_dataset(self, train_dataset, val_dataset):
        """
        Convert Q&A JSON into formatted text prompts.
        Adds EOS tokens and formats data with optional system prompt.
        """
        # Get prompt from config (or use default)
        system_prompt = self.cfg.get("prompt", "Answer the following question accurately.")
        
        # Build the full prompt template
        qa_prompt = f"""{system_prompt}
        Question: {{}}

        Answer: {{}}"""
        
        eos = self.tokenizer.eos_token
        
        def formatting_prompts_func(examples):
            # Your data has "question" and "answer" fields
            questions = examples.get("question", [])
            answers = examples.get("answer", [])
            
            texts = []
            for question, answer in zip(questions, answers):
                text = qa_prompt.format(question, answer) + eos
                texts.append(text)
            
            return {"text": texts}
        
        print("🧩 Formatting datasets...")
        if "prompt" in self.cfg:
            print(f"   Using custom prompt: {self.cfg['prompt'][:60]}...")
        else:
            print("   Using default prompt")
        
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
        print("✅ Datasets formatted successfully")
        return train_dataset, val_dataset



    # --------------------------------------------------------------------
    def train(self):
        """
        Launch QLoRA fine-tuning using TRL’s SFTTrainer.

        Handles:
        - gradient accumulation
        - validation every N steps
        - cosine LR schedule
        - mixed precision (fp16 / bf16)
        """
        print("🏋️ Starting QLoRA training...")

        # Load and format data
        train_dataset, val_dataset = self.load_datasets()
        train_dataset, val_dataset = self.format_dataset(train_dataset, val_dataset)

        # ---- Training configuration ----
        training_args = TrainingArguments(
            per_device_train_batch_size=self.cfg.get("batch_size", 2),
            per_device_eval_batch_size=self.cfg.get("eval_batch_size", 2),
            gradient_accumulation_steps=self.cfg.get("grad_accum_steps", 4),
            warmup_steps=self.cfg.get("warmup_steps", 10),
            num_train_epochs=self.cfg.get("epochs", 3),
            learning_rate=self.cfg.get("lr", 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="paged_adamw_8bit",  # memory efficient optimizer
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=str(self.save_path / "outputs"),
            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # ---- Create trainer ----
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # disable sample packing
            args=training_args,
        )

        # ---- Train ----
        trainer_stats = trainer.train()
        print("✅ Training completed successfully.")

        # ---- Save model ----
        self.save(trainer_stats)

    # --------------------------------------------------------------------
    def save(self, trainer_stats):
        """
        Save LoRA adapter weights and tokenizer to configured save path.
        The result is a lightweight adapter folder (≈100MB vs full 7B model).
        """
        print("💾 Saving LoRA adapter weights...")
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print(f"✅ Weights saved to {self.save_path}")

    # --------------------------------------------------------------------
    def merge_adapters(self):
        """
        (Optional and currently not use. Use if you are planning to use VLLM)
        Merge LoRA adapter weights into the base model to produce a full,
        standalone model for deployment / inference without PEFT.
        """
        print("🔄 Merging LoRA adapters into base model...")
        merged_model = FastLanguageModel.merge_and_unload(self.model)
        merged_path = self.save_path / "merged_full_model"
        merged_model.save_pretrained(merged_path)
        self.tokenizer.save_pretrained(merged_path)
        print(f"✅ Merged model saved to {merged_path}")


# --------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a QLoRA model with Unsloth.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    # Load configuration JSON (e.g. car_config.json)
    with open(args.config, "r") as f:
        config = json.load(f)

    # Initialize trainer
    trainer = QLoRATrainer(config)

    # Attach LoRA adapters
    trainer.prepare_lora()

    # Start fine-tuning
    trainer.train()

    # Optionally merge adapters after training if using vLLMs
    # trainer.merge_adapters()
