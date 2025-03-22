import torch._dynamo
torch._dynamo.config.suppress_errors = True


import torch
from transformers import AutoTokenizer
from transformers.models.gemma import GemmaForCausalLM

model_id = "shuyuej/gemma-2b-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GemmaForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "What is the highest montain in korea?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

import time, math

with torch.no_grad():
    labels = inputs["input_ids"]
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    perplexity = math.exp(loss.item())

start_time = time.time()
generate_ids = model.generate(**inputs, max_new_tokens=30)
end_time = time.time()

print("=== GPTQ INT4 모델 ===")
print(tokenizer.decode(generate_ids[0], skip_special_tokens=True))
print(f"⏱️ 추론 시간: {end_time - start_time:.2f}초")
print(f"🔍 Perplexity: {perplexity:.2f}")
