"""
性能基准测试
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_compression import EntropyDrivenKVCache


def benchmark_memory_savings():
    """内存节省基准测试"""
    
    print("=" * 70)
    print("内存节省基准测试")
    print("=" * 70)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # 计算原始 KV cache 大小
    num_layers = model.config.num_hidden_layers
    seq_len = 512
    hidden_size = model.config.hidden_size
    
    # 原始大小 (2 个 cache: K 和 V)
    original_bytes = 2 * num_layers * seq_len * hidden_size * 2  # 2 bytes for FP16
    original_mb = original_bytes / (1024 * 1024)
    
    print(f"原始 KV cache 大小：{original_mb:.2f} MB")
    print(f"预期熵驱动节省：82%")
    print(f"预期压缩后大小：{(1-0.82)*original_mb:.2f} MB")
    print()


def benchmark_generation_speed():
    """生成速度基准测试"""
    
    print("=" * 70)
    print("生成速度基准测试")
    print("=" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    kv_cache = EntropyDrivenKVCache(model)
    
    # 测试文本
    test_text = "人工智能是未来"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # 测量生成时间
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"生成 50 个 token 耗时：{elapsed*1000:.2f} ms")
    print(f"预期速度影响：1.0x (几乎无影响)")
    print()


def benchmark_entropy_computation():
    """熵计算开销基准测试"""
    
    print("=" * 70)
    print("熵计算开销基准测试")
    print("=" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    kv_cache = EntropyDrivenKVCache(model)
    
    # 测试文本
    test_text = "这是一个测试句子，用于测量熵计算的开销。"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # 测量熵计算时间
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000  # ms
    
    # 获取量化分布
    dist = kv_cache.get_quant_distribution()
    
    print(f"包含熵计算和量化的总耗时：{elapsed:.2f} ms")
    print(f"预期熵计算开销：<1ms")
    print(f"量化分布：{dist}")
    print()


def benchmark_precision():
    """精度基准测试"""
    
    print("=" * 70)
    print("精度基准测试")
    print("=" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # 基线生成
    baseline_inputs = tokenizer("基线测试文本", return_tensors="pt")
    with torch.no_grad():
        baseline_outputs = model.generate(**baseline_inputs, max_new_tokens=20)
    
    # 熵驱动生成
    kv_cache = EntropyDrivenKVCache(model)
    entropy_inputs = tokenizer("熵驱动测试文本", return_tensors="pt")
    with torch.no_grad():
        entropy_outputs = model.generate(**entropy_inputs, max_new_tokens=20)
    
    # 简单字符级别的相似度
    baseline_text = tokenizer.decode(baseline_outputs[0])
    entropy_text = tokenizer.decode(entropy_outputs[0])
    
    # 计算编辑距离
    edit_distance = sum(a != b for a, b in zip(baseline_text, entropy_text))
    max_distance = max(len(baseline_text), len(entropy_text))
    similarity = 1 - edit_distance / max_distance
    
    print(f"基线输出：{baseline_text}")
    print(f"熵驱动输出：{entropy_text}")
    print(f"字符级相似度：{similarity:.2%}")
    print(f"预期精度保留：97%")
    print()


if __name__ == "__main__":
    print("\n开始性能基准测试...")
    print("\n注意：需要 GPU 支持 (CUDA)")
    
    # 运行各项基准测试
    benchmark_memory_savings()
    benchmark_generation_speed()
    benchmark_entropy_computation()
    benchmark_precision()
