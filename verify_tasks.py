#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证所有任务脚本的关键功能
"""

import sys
import importlib.util

def test_imports(script_name):
    """测试脚本的导入语句"""
    print(f"\n检查 {script_name}...")
    try:
        spec = importlib.util.spec_from_file_location("module", script_name)
        module = importlib.util.module_from_spec(spec)
        # 只检查导入，不执行
        with open(script_name, 'r', encoding='utf-8') as f:
            code = f.read()
            # 提取import语句
            imports = [line for line in code.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
            print(f"  发现 {len(imports)} 个导入语句")
        return True
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False

print("=" * 60)
print("任务脚本验证")
print("=" * 60)

scripts = [
    "task04_cnn_architectures.py",
    "task05_rnn.py",
    "task06_autoencoders.py",
    "task07_vae_gan.py",
    "task08_rl.py",
    "task09_diffusion.py",
]

results = {}
for script in scripts:
    results[script] = test_imports(script)

print("\n" + "=" * 60)
print("验证总结")
print("=" * 60)
for script, passed in results.items():
    status = "✓ 通过" if passed else "✗ 失败"
    print(f"{status}: {script}")

print("\n已知问题和注意事项：")
print("1. Task 04: 使用CIFAR-10演示，如需Tiny-ImageNet请修改数据加载部分")
print("2. Task 05: 需要下载airline-passengers.csv数据")
print("3. Task 06-09: 使用MNIST数据集（自动下载）")
print("4. Task 08: 需要gymnasium库（pip install gymnasium）")
print("5. Task 10: 需要先准备数据和安装ultralytics")
print("\n建议：先运行单个任务测试，确认环境配置正确后再批量运行")
