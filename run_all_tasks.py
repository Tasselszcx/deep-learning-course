#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键运行所有深度学习课程任务
"""

import subprocess
import sys

tasks = [
    ("Task 04: CNN架构", "task04_cnn_architectures.py"),
    ("Task 05: RNN", "task05_rnn.py"),
    ("Task 06: Autoencoders", "task06_autoencoders.py"),
    ("Task 07: VAE & GAN", "task07_vae_gan.py"),
    ("Task 08: 强化学习", "task08_rl.py"),
    ("Task 09: 扩散模型", "task09_diffusion.py"),
]

print("=" * 60)
print("深度学习课程任务自动运行脚本")
print("=" * 60)

for i, (name, script) in enumerate(tasks, 1):
    print(f"\n[{i}/{len(tasks)}] 运行 {name}...")
    print("-" * 60)

    try:
        result = subprocess.run([sys.executable, script],
                              capture_output=False,
                              text=True)
        if result.returncode == 0:
            print(f"✓ {name} 完成")
        else:
            print(f"✗ {name} 失败")
    except Exception as e:
        print(f"✗ {name} 出错: {e}")

print("\n" + "=" * 60)
print("所有任务运行完成！")
print("=" * 60)
print("\n注意：Task 10 (YOLO) 需要先准备数据，请单独运行 task10_yolo.py")
