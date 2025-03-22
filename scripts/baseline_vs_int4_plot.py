import matplotlib.pyplot as plt

# 결과 데이터 (여기에 직접 입력!)
models = ["Baseline", "INT4"]
perplexities = [30.80, 109.82]  # 예시: 실제 baseline 결과로 수정
latencies = [51.91, 6.38]        # 예시: 실제 baseline 결과로 수정

# 시각화
plt.figure(figsize=(10, 5))

# 🔵 추론 시간 그래프
plt.subplot(1, 2, 1)
plt.bar(models, latencies)
plt.title("Inference Time (s)")
plt.ylabel("Seconds")

# 🔴 Perplexity 그래프
plt.subplot(1, 2, 2)
plt.bar(models, perplexities, color='orange')
plt.title("Perplexity")
plt.ylabel("Score")

plt.tight_layout()
plt.show()
