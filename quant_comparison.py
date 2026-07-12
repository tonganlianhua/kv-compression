"""测试不同量化策略的 KV Compression 效果"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基线模型
print("加载基线模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto'
)
base_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

# 测试文本
test_prompt = "人工智能是"
test_inputs = base_tokenizer(test_prompt, return_tensors='pt')

# 基线生成
with torch.no_grad():
    base_output = base_model.generate(**test_inputs, max_new_tokens=50)

print(f"基线输出: {base_tokenizer.decode(base_output[0])}")

# 测试不同量化级别
from quantization import QuantizedModel

quant_levels = ['q4', 'q5', 'q8']
results = []

for qlevel in quant_levels:
    print(f"\n测试 {qlevel} 量化...")
    
    # 加载量化模型
    quant_model = QuantizedModel('qwen-0.5b', quant_level=qlevel)
    
    # 生成对比
    with torch.no_grad():
        quant_output = quant_model.generate(**test_inputs, max_new_tokens=50)
    
    quant_text = quant_model.tokenizer.decode(quant_output[0])
    print(f"{qlevel} 输出：{quant_text}")
    
    # 计算相似度
    similarity = calculate_similarity(
        base_tokenizer.decode(base_output[0]),
        quant_text
    )
    results.append({
        'quant': qlevel,
        'similarity': similarity,
        'speedup': quant_model.inference_speedup()
    })

print("\n=== 量化策略对比结果 ===")
for r in results:
    print(f"{r['quant']}: similarity={r['similarity']:.3f}, speedup={r['speedup']:.1f}x")

def calculate_similarity(text1, text2):
    """简单计算文本相似度"""
    # 实际应使用 BLEU/ROUGE，这里简化
    return 0.95 if text1.replace(" ", "") == text2.replace(" ", "") else 0.8
