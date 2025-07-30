from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load PEFT config
peft_model_id = "tinyllama-qlora/peft"  # or checkpoint-xxx if using intermediate
config = PeftConfig.from_pretrained(peft_model_id)

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",  # or "cpu"
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load LoRA/QLoRA adapter weights
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

# Inference
input_text = "Tell me a joke about AI."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
