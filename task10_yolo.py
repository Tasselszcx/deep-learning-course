# Task 10: YOLO屋顶分类
# 注意：此任务需要收集数据和训练YOLO模型

"""
Task 10 实施指南：

1. 数据收集：
   - 从Google Earth、OpenAerialMap等来源收集屋顶航拍图像
   - 分类为 flat（平顶）和 pitched（斜顶）
   - 组织数据结构：
     aerial/
     ├─ train/
     │  ├─ flat/
     │  └─ pitched/
     ├─ val/
     │  ├─ flat/
     │  └─ pitched/
     └─ test/
        ├─ flat/
        └─ pitched/

2. 训练代码：
"""

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# 配置
DATA_ROOT = Path("./aerial")  # 数据路径
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# 加载预训练模型
model = YOLO("yolov8n-cls.pt")

# 训练参数（可调整以提高性能）
results = model.train(
    data=str(DATA_ROOT),
    imgsz=320,              # 图像大小：224/320/384
    epochs=30,              # 训练轮次
    batch=64,               # 批次大小
    device=DEVICE,
    workers=2,
    patience=10,
    auto_augment="randaugment",  # 数据增强
    erasing=0.5,
    mixup=0.1,
    cutmix=0.1,
    verbose=True,
    plots=True
)

# 获取最佳模型路径
best_ckpt = model.trainer.best
print(f"最佳模型: {best_ckpt}")

# 在测试集上评估
metrics = model.val(
    data=str(DATA_ROOT),
    split="test",
    imgsz=320,
    device=DEVICE
)

print(f"测试准确率: {100*metrics.top1:.2f}%")

# 保存模型（替换CID为你的学号）
import shutil
CID = "your_college_id"  # 替换为你的学号
dst = Path(f"./{CID}_best.pt")
shutil.copy(best_ckpt, dst)
print(f"模型已保存: {dst}")

# 绘制学习曲线
# YOLO会自动生成训练曲线，位于 runs/classify/train/results.png

print("""
Task 10 完成步骤：
1. 收集并标注屋顶图像数据
2. 组织数据到 aerial/ 目录
3. 运行此脚本训练模型
4. 提交 {CID}_best.pt 和 {CID}_data.zip
5. 在报告中包含：
   - 分类性能
   - 训练数据详情
   - 训练参数
   - 学习曲线图
""")
