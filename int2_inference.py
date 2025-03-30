import torch
import time
import os
import json
import math
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# 1. 모델 경로 및 장치 설정
model_path = "gemma-2b-INT2-GPTQ"
device = "cpu"  # ✅ CPU 고정

# 2. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoGPTQForCausalLM.from_quantized(model_path, device="cpu", use_triton=False)

# 3. 예시 프롬프트
prompt = "Explain how a black hole is formed in space."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 4. 추론 수행
start_time = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
end_time = time.time()

print("\n▶ [Generated Output]")
print(tokenizer.decode(output[0], skip_special_tokens=True))
print(f"\n⏱ Inference time: {end_time - start_time:.2f} seconds")

# 5. Perplexity 계산 함수
def calculate_perplexity(model, tokenizer, max_samples=20):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1%]")
    total_loss = 0.0
    total_length = 0

    model.eval()
    for example in tqdm(dataset.select(range(max_samples)), desc="Calculating PPL"):
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        total_loss += loss.item() * inputs["input_ids"].size(1)
        total_length += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_length
    ppl = math.exp(avg_loss)
    return ppl

# 6. 퍼플렉서티 출력
ppl = calculate_perplexity(model, tokenizer)
print(f"\n📊 Perplexity (INT2 on CPU): {ppl:.2f}")
