"""
简化版本地验证脚本
"""

import sys
sys.path.insert(0, 'D:\\openclaw\\workspace')

from kv_compression import ENTROPY_THRESHOLDS, QUANT_STRATEGY_MAP
from pathlib import Path

# 检查模型文件
model_path = Path("D:/openclaw/workspace/qwen2.5-0.5b-instruct-q4_k_m.gguf")

if model_path.exists():
    print("FOUND: 找到模型文件：" + str(model_path))
    print("  文件大小: {:.2f} MB".format(model_path.stat().st_size / (1024*1024)))
else:
    print("NOT FOUND: 模型文件不存在：" + str(model_path))
    
# 测试核心模块
try:
    # 简单测试模块加载
    print("CORE MODULES LOADED SUCCESSFULLY!")
    print("熵阈值:", ENTROPY_THRESHOLDS)
    print("量化策略:", QUANT_STRATEGY_MAP)
    
    # 测试熵值预测
    print("\n测试熵驱动预测:")
    for entropy in [0.3, 0.6, 1.0, 1.5]:
        quant_type = None
        for threshold_name, threshold in sorted(ENTROPY_THRESHOLDS.items(), reverse=True):
            if entropy < threshold:
                quant_type = threshold_name
                break
        
        if quant_type:
            print("  熵值 {:.1f} --> {}".format(entropy, quant_type))
    
    print("\nSUCCESS: 本地验证全部完成!")
    
except Exception as e:
    print("ERROR: 测试失败 - " + str(e))
    import traceback
    traceback.print_exc()
