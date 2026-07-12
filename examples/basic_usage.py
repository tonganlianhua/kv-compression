"""
基础使用示例
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_compression import EntropyDrivenKVCache
import torch


def example_basic_usage():
    """基础使用示例"""
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # 创建熵驱动 KV Cache
    kv_cache = EntropyDrivenKVCache(model)
    
    # 生成文本
    text = "人工智能是未来"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    print(tokenizer.decode(outputs[0]))
    
    # 查看量化分布
    dist = kv_cache.get_quant_distribution()
    print(f"量化分布：{dist}")


def example_entropy_tracking():
    """熵追踪示例"""
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    kv_cache = EntropyDrivenKVCache(model)
    
    # 测试不同文本
    test_texts = [
        "简单文本",
        "复杂的科学技术术语",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        
        dist = kv_cache.get_quant_distribution()
        print(f"文本：{text[:30]}")
        print(f"量化分布：{dist}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("基础使用示例")
    print("=" * 60)
    
    # 运行基础示例
    print("\n1. 基础生成:")
    example_basic_usage()
    
    # 运行熵追踪示例
    print("\n2. 熵追踪:")
    example_entropy_tracking()
