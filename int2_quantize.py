from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch
import os
import json

model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 1. 안정성 강화된 수동 양자화 설정 클래스
class QuantConfig:
    bits = 2
    group_size = 128
    desc_act = False
    sym = False
    true_sequential = False
    damp_percent = 0.05
    static_groups = False
    model_name_or_path = model_id
    model_file_base_name = "model"
    is_marlin_format = False

    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "damp_percent": self.damp_percent,
            "static_groups": self.static_groups,
            "model_name_or_path": self.model_name_or_path,
            "model_file_base_name": self.model_file_base_name,
            "is_marlin_format": self.is_marlin_format,
        }

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "quantize_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

# 2. 양자화 설정 객체 생성
quant_config = QuantConfig()

# 3. 모델 로드
model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quant_config,
    device_map="auto"
)

# 4. 다양한 예시 문장 (길이+주제 다양)
prompts = [
    "What is the capital of France and what is its population and cultural heritage?",
    "Explain the theory of relativity in detail with practical examples and implications.",
    "List the major mountains in Korea and explain their geographical significance.",
    "Describe the process of training a large language model step-by-step including optimization strategies.",
    "Summarize the impact of climate change on global agriculture and food systems.",
    "How do neural networks differ from traditional machine learning methods?",
    "What are the advantages and disadvantages of solar power in modern energy systems?",
    "Compare the philosophies of Confucianism and Buddhism in East Asian history.",
    "Explain how blockchain technology works with a detailed example.",
    "Discuss the economic impact of the pandemic on global trade routes."
]
examples = [tokenizer(p, return_tensors="pt") for p in prompts]

# 5. 양자화 실행
model.quantize(examples)

# 6. 모델 및 토크나이저 저장
model.save_quantized("gemma-2b-INT2-GPTQ")
tokenizer.save_pretrained("gemma-2b-INT2-GPTQ")
