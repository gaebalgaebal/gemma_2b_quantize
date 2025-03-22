import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import math

model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

prompt = "What is the highest mountain in Korea?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Perplexity 계산용
with torch.no_grad():
    labels = inputs["input_ids"]
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    perplexity = math.exp(loss.item())

# 추론
start_time = time.time()
generate_ids = model.generate(**inputs, max_new_tokens=30)
end_time = time.time()

print("=== Baseline ===")
print(tokenizer.decode(generate_ids[0], skip_special_tokens=True))
print(f"⏱️ 추론 시간: {end_time - start_time:.2f}초")
print(f"🔍 Perplexity: {perplexity:.2f}")
