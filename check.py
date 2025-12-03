import torch

print("=" * 70)
print("GPU 检测报告")
print("=" * 70)

# 1. CUDA 是否可用
print(f"\n1. CUDA 是否可用: {torch.cuda.is_available()}")

# 2. PyTorch 版本
print(f"2. PyTorch 版本: {torch.__version__}")

# 3. CUDA 版本
if torch.cuda.is_available():
    print(f"3. CUDA 版本: {torch.version.cuda}")
    print(f"4. GPU 数量: {torch.cuda.device_count()}")
    print(f"5. 当前 GPU: {torch.cuda.current_device()}")
    print(f"6. GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"7. GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 测试 GPU
    print("\n测试 GPU 计算:")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print(f"  ✓ GPU 计算成功!")
    print(f"  结果设备: {z.device}")
else:
    print("\n❌ CUDA 不可用!")
    print("可能原因:")
    print("  1. 没有 NVIDIA GPU")
    print("  2. 安装的是 CPU 版本的 PyTorch")
    print("  3. CUDA 驱动未安装")

print("=" * 70)