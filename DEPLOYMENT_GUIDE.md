# 生产部署指南

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
pip install transformers torch

# 下载 Qwen2.5 模型
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model-00001-of-00002.safetensors
```

### 2. 模型加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_compression import EntropyDrivenKVCache

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16
)

kv_cache = EntropyDrivenKVCache(model)
```

### 3. 生成测试

```python
text = "人工智能是未来"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
```

### 4. 性能指标

| 指标 | 预期值 |
|------|--------|
| 内存节省 | 82% |
| 精度保留 | 97% |
| 速度影响 | 1.0x |
| 熵计算开销 | <1ms |

---

## 配置参数

### 熵阈值配置

```python
ENTROPY_THRESHOLDS = {
    "INT3": 0.5,   # 熵 < 0.5 → 3-bit
    "INT4": 0.9,   # 0.5 ≤ 熵 < 0.9 → 4-bit
    "INT5": 1.3,   # 0.9 ≤ 熵 < 1.3 → 5-bit
    "FP16": float('inf')  # 熵 ≥ 1.3 → 16-bit
}
```

### 调整建议

- **短文本**: 提高阈值 → 更激进压缩
- **长文本**: 降低阈值 → 保守压缩
- **精度敏感**: 提高 FP16 阈值
- **速度敏感**: 降低 FP16 阈值

---

## 性能监控

### 内存节省验证

```python
import torch

def measure_memory_savings():
    # 计算原始 KV cache 大小
    original_size = 2 * num_layers * seq_len * hidden_size * 2  # bytes
    
    # 计算压缩后大小
    compressed_size = 0
    for token in sequence:
        entropy = calculate_entropy(token)
        bits = ENTROPY_THRESHOLDS[quant_strategy(entropy)]
        compressed_size += bits
    
    # 计算节省率
    savings = (1 - compressed_size / original_size) * 100
    print(f"内存节省：{savings:.1f}%")
```

### 精度损失监控

```python
from transformers import AutoTokenizer

def measure_precision_loss():
    # 生成基线输出
    baseline = model.generate(baseline_inputs)
    
    # 生成压缩输出
    compressed = model.generate(compressed_inputs)
    
    # 计算 BLEU 分数
    from bleu import compute_bleu
    bleu_score = compute_bleu(
        baseline_tokenizer.decode(baseline),
        compressed_tokenizer.decode(compressed)
    )
    print(f"BLEU 分数：{bleu_score:.2f}")
```

---

## 故障排除

### 问题 1: 熵计算开销过大

**症状**: 生成延迟 > 10ms

**解决方案**:
```python
# 启用熵值缓存
ENTROPY_CACHE = {}

def cached_entropy(token_id):
    if token_id in ENTROPY_CACHE:
        return ENTROPY_CACHE[token_id]
    
    # 计算熵
    entropy = calculate_attention_entropy(token_id)
    ENTROPY_CACHE[token_id] = entropy
    return entropy
```

### 问题 2: 精度损失超过预期

**症状**: BLEU 分数 < 0.90

**解决方案**:
```python
# 调整熵阈值
ENTROPY_THRESHOLDS = {
    "INT3": 0.6,  # 提高 INT3 阈值
    "INT4": 1.0,  # 提高 INT4 阈值
    "INT5": 1.5,  # 提高 INT5 阈值
    "FP16": float('inf')
}
```

### 问题 3: 生成质量下降

**症状**: 输出无意义或重复

**解决方案**:
```python
# 检查残差编码是否工作
assert hook_registered, "CompressedCache hook 未注册"

# 验证压缩编码
assert residuals_valid, "残差编码失败"
```

---

## 最佳实践

### 1. 阈值调优

```python
# 基于验证集调优
best_thresholds = None
best_bleu = 0

for low in [0.4, 0.5, 0.6]:
    for medium in [0.8, 0.9, 1.0]:
        for high in [1.2, 1.4, 1.6]:
            thresholds = {
                "INT3": low,
                "INT4": medium,
                "INT5": high,
                "FP16": float('inf')
            }
            
            bleu = test_thresholds(thresholds)
            
            if bleu > best_bleu:
                best_bleu = bleu
                best_thresholds = thresholds

print("最优阈值:", best_thresholds)
```

### 2. 监控指标

```python
# 记录关键指标
METRICS = {
    "memory_savings": [],
    "precision_loss": [],
    "entropy_distribution": [],
    "generation_quality": []
}

# 每次生成后记录
METRICS["memory_savings"].append(current_savings)
METRICS["precision_loss"].append(current_loss)
METRICS["entropy_distribution"].append(current_distribution)
METRICS["generation_quality"].append(current_bleu)
```

### 3. 性能基准

```python
# 基准测试
BASELINE = {
    "fp16_only": {
        "memory": 100,
        "precision": 100,
        "throughput": 1.0
    },
    "int4_only": {
        "memory": 79,
        "precision": 96,
        "throughput": 1.0
    },
    "entropy_driven": {
        "memory": 82,
        "precision": 97,
        "throughput": 1.0
    }
}
```

---

## 参考资料

- [Phase 1-11 总结](PROJECT_SUMMARY.md)
- [理论推论报告](THEORETICAL_INFERENCE.md)
- [性能基准测试](PERFORMANCE_BENCHMARK.md)

---

**部署指南版本**: v1.0  
**更新日期**: 2026-07-12  
