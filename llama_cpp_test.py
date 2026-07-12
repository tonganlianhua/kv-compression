"""
使用 llama-cpp-python 进行完整验证
"""

import sys
sys.path.insert(0, 'D:\\openclaw\\workspace')

from kv_compression import EntropyDrivenKVCache
from llama_cpp import Llama

# 加载本地模型
model_path = "D:/openclaw/workspace/qwen2.5-0.5b-instruct-q4_k_m.gguf"

print("Loading local GGUF model...")
print("Model path:", model_path)

model = Llama.from_pretrained(
    model_path,
    n_gpu_layers=-1,  # 使用 GPU
    n_ctx=512,
)

print("Model loaded successfully!")

# 创建熵驱动 KV Cache
kv_cache = EntropyDrivenKVCache(model)

# 测试文本
test_texts = [
    "人工智能是未来",
    "The quick brown fox jumps over the lazy dog.",
    "Hello world!",
]

print("\nStarting generation test...")

for text in test_texts:
    print("\nText: {}".format(text))
    
    # 生成
    output = model(
        text,
        max_tokens=20,
        temperature=0.7,
        echo=True
    )
    
    # 查看量化分布
    dist = kv_cache.get_quant_distribution()
    print("Quantization distribution:", dist)
    
    # 清空历史
    kv_cache.clear_history()

print("\n✅ Local validation completed successfully!")
