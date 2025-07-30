from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./unsloth_tinyllama-qlora/peft",
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)  # Puts it in eval mode & disables dropout

inputs = tokenizer("Who won the 2024 olympics?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
