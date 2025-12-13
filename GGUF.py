from unsloth import FastLanguageModel

# 1. Load your local fine-tuned model
print("Loading local model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "llama32_3b_uncensored_final", # Your local folder
    max_seq_length=8192,
    load_in_4bit=True,
)

# 2. Convert to GGUF (Quantized 4-bit Medium)
# This merges the LoRA into the Base Model and saves it as one file.
print("Converting to GGUF (q4_k_m)...")
model.save_pretrained_gguf("Aletheia-GGUF", tokenizer, quantization_method = "q4_k_m")

print("✅ Conversion Complete. File is in 'Aletheia-GGUF' folder.")