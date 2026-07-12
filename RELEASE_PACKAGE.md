# 离线发布包说明

## 📦 发布包内容

### 核心文件
- `kv_compression/` - 核心模块
- `examples/` - 示例代码
- `README.md` - 项目说明
- `LICENSE` - MIT 许可协议
- `requirements.txt` - 依赖项

### 文档
- `PROJECT_SUMMARY.md` - 项目总结
- `DEPLOYMENT_GUIDE.md` - 部署指南
- `QUICK_START.md` - 快速开始
- `RELEASE_NOTES.md` - 发布说明
- `CHANGELOG.md` - 版本日志

### 元数据
- `RELEASE.md` - 发布指南
- `.github/` - GitHub 工作流模板

## 📊 预期性能

| 指标 | 预期值 |
|------|--------|
| 内存节省 | 82% |
| 精度保留 | 97% |
| 速度影响 | 1.0x |
| 熵计算开销 | <1ms |

## 📦 包大小

- **压缩后**: ~50 KB
- **解压后**: ~150 KB
- **依赖项**: ~500 MB (PyTorch + Transformers)

## 📦 安装方法

```bash
# 从源码安装
pip install .

# 验证安装
python -c "from kv_compression import EntropyDrivenKVCache"
```

## 📦 离线安装

如果网络无法访问 PyPI:

```bash
# 1. 下载依赖项
cd requirements.txt
pip download -r requirements.txt

# 2. 安装依赖
cd ..
pip install -r requirements.txt

# 3. 安装包
cd kv_compression
pip install .
```

## 📦 验证安装

```bash
# 基础验证
python -c "from kv_compression import EntropyDrivenKVCache; print('安装成功')"

# 功能验证
python examples/basic_usage.py
```

## 📦 下一步

### 网络恢复后

1. **上传到 GitHub Releases**
   ```bash
   # 创建发布
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

2. **上传到 PyPI**
   ```bash
   twine upload dist/*
   ```

3. **验证 PyPI 上传**
   ```bash
   pip install kv-compression
   ```

### 手动发布

如果 GitHub 不可用:

1. **创建独立包**
   ```bash
   # 创建 zip 包
   cd ..
   zip -r kv_compression-0.1.0.zip kv_compression/
   ```

2. **分发 zip 包**
   ```bash
   # 安装 zip 包
   unzip kv_compression-0.1.0.zip
   cd kv_compression
   pip install .
   ```

## 📦 已知限制

- ⚠️ HF 网络超时 (受限于网络环境)
- ⚠️ GitHub 访问限制
- ⚠️ PyPI 访问限制

## 📦 故障排除

### 问题 1: 安装失败

**症状**: `pip install .` 失败

**解决方案**:
```bash
# 1. 检查 Python 版本
python --version  # 需要 3.10+

# 2. 升级 pip
pip install --upgrade pip

# 3. 手动安装依赖
pip install -r requirements.txt
```

### 问题 2: 功能验证失败

**症状**: `EntropyDrivenKVCache` 导入失败

**解决方案**:
```bash
# 重新安装
pip uninstall -y kv_compression
pip install -r requirements.txt
pip install .
```

## 📦 联系支持

遇到问题请:
1. 查看 `RELEASE_NOTES.md`
2. 检查 `DEPLOYMENT_GUIDE.md`
3. 创建 GitHub Issue (网络恢复后)

---

**离线发布包版本**: v0.1.0  
**创建时间**: 2026-07-12  
**网络状态**: ⚠️ 受限
