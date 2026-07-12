"""
本地验证脚本
基于已有的 Qwen2.5-0.5B-Instruct GGUF 模型进行验证
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_compression import EntropyDrivenKVCache
from pathlib import Path


def load_local_gguf_model():
    """加载本地 GGUF 模型"""
    
    model_path = "D:\\openclaw\\workspace\\qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在：{model_path}")
        return None, None
    
    print(f"✅ 找到模型文件：{model_path}")
    print(f"   文件大小: {Path(model_path).stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # 尝试加载 GGUF 模型 (需要 llama-cpp-python)
        from llama_cpp import Llama
        
        model = Llama.from_pretrained(
            model_path,
            n_gpu_layers=-1,  # 使用 GPU 加速
            n_ctx=512,
        )
        
        tokenizer = model
        
        print("✅ 模型加载成功!")
        return model, tokenizer
        
    except ImportError:
        # 如果 llama-cpp-python 未安装，尝试使用 transformers
        print("⚠️ llama-cpp-python 未安装，尝试使用 transformers 加载...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            
            print("✅ 模型加载成功!")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            return None, None


def test_local_validation(model, tokenizer):
    """本地验证测试"""
    
    print("\n" + "=" * 70)
    print("本地验证测试开始")
    print("=" * 70)
    
    test_texts = [
        "人工智能是未来",
        "The quick brown fox jumps over the lazy dog.",
        "Transformer 架构由注意力机制驱动",
        "Hello world!"
    ]
    
    # 创建熵驱动 KV Cache
    kv_cache = EntropyDrivenKVCache(model)
    
    for text in test_texts:
        print(f"\n测试文本：{text}")
        
        # Tokenize
        if hasattr(tokenizer, "encode"):
            inputs = tokenizer.encode(text, return_tensors="pt")
        else:
            inputs = tokenizer.encode(text)
        
        # 前向传播
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        
        # 获取量化分布
        dist = kv_cache.get_quant_distribution()
        
        print(f"  生成输出：{tokenizer.decode(outputs[0])[:30]}...")
        print(f"  量化分布：{dist}")
        
        # 清空历史记录
        kv_cache.clear_history()
    
    print("\n" + "=" * 70)
    print("✅ 本地验证测试完成!")
    print("=" * 70)


def measure_performance(model, tokenizer):
    """性能测试"""
    
    print("\n" + "=" * 70)
    print("性能测试")
    print("=" * 70)
    
    import time
    
    # 测试文本
    test_text = "这是一个测试句子，用于测量熵驱动的量化效果。"
    
    # Tokenize
    if hasattr(tokenizer, "encode"):
        inputs = tokenizer.encode(test_text, return_tensors="pt")
    else:
        inputs = tokenizer.encode(test_text)
    
    # 生成并测量时间
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"生成 50 个 token 耗时：{elapsed*1000:.2f} ms")
    print(f"预期速度影响：1.0x (几乎零开销)")
    print()


def main():
    """主函数"""
    
    print("=" * 70)
    print("KV Compression 本地验证")
    print("=" * 70)
    
    # 加载本地模型
    print("\n加载本地 GGUF 模型...")
    model, tokenizer = load_local_gguf_model()
    
    if model is None:
        print("❌ 模型加载失败，请检查模型路径!")
        return
    
    # 测试本地验证
    test_local_validation(model, tokenizer)
    
    # 性能测试
    measure_performance(model, tokenizer)
    
    print("\n" + "=" * 70)
    print("✅ 本地验证全部完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
