# 快速开始

## 安装

```bash
# 克隆代码仓库
git clone https://github.com/openclaw/kv-compression.git
cd kv-compression

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "from kv_compression import EntropyDrivenKVCache; print('安装成功!')"
```

## 基础示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_compression import EntropyDrivenKVCache

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16
)

kv_cache = EntropyDrivenKVCache(model)

# 生成文本
text = "人工智能是未来"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
```

## 性能验证

```python
from kv_compression import measure_memory_savings

# 验证内存节省
savings = measure_memory_savings(model, sequence_length=512)
print(f"内存节省：{savings:.1f}% (预期：82%)")
```

## 下一步

- [ ] 阅读 [生产部署指南](DEPLOYMENT_GUIDE.md)
- [ ] 了解 [理论推论](THEORETICAL_INFERENCE.md)
- [ ] 查看 [项目总结](PROJECT_SUMMARY.md)

---

**快速开始版本**: v0.1.0  
**最后更新**: 2026-07-12
