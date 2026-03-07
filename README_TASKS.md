# 深度学习课程任务04-10完成指南

## 概述

我已经为你创建了7个Python脚本，每个脚本对应一个任务。这些脚本包含了完成所有coursework任务所需的核心代码。

## 已创建的文件

1. **task04_cnn_architectures.py** - Task 04: CNN架构对比
2. **task05_rnn.py** - Task 05: RNN时间序列预测
3. **task06_autoencoders.py** - Task 06: 自编码器表示学习
4. **task07_vae_gan.py** - Task 07: VAE和GAN生成模型
5. **task08_rl.py** - Task 08: 强化学习（Q-learning vs SARSA）
6. **task09_diffusion.py** - Task 09: 扩散模型
7. **task10_yolo.py** - Task 10: YOLO屋顶分类

## 运行顺序

按照以下顺序运行脚本以生成所有需要的结果：

```bash
# Task 04
python task04_cnn_architectures.py

# Task 05
python task05_rnn.py

# Task 06
python task06_autoencoders.py

# Task 07
python task07_vae_gan.py

# Task 08
python task08_rl.py

# Task 09
python task09_diffusion.py

# Task 10 (需要先准备数据)
python task10_yolo.py
```

## 每个任务的输出

### Task 04: Common CNN Architectures
**输出文件：**
- `Task4_1_accuracy_curves.png` - 训练和验证准确率曲线
- `Task4_1_results_table.csv` - 性能对比表格
- `Task4_2_test_accuracy.png` - 测试集准确率对比

**报告内容：**
- Task 4.1: 图表 + 表格
- Task 4.2: 图表

### Task 05: RNN
**输出文件：**
- `Task5_1_predictions.png` - 不同窗口大小的预测曲线
- `Task5_2_results.csv` - 性能对比表格
- `Task5_3_best_model.png` - 最佳模型预测结果

**报告内容：**
- Task 5.1: 图表
- Task 5.2: 表格（附录中的图表）
- Task 5.3: 图表

### Task 06: Autoencoders
**输出文件：**
- `Task6_1_results.csv` - 分类准确率对比表格
- `Task6_2_reconstruction.csv` - 重构质量表格
- `Task6_2_reconstruction_visual.png` - 重构图像可视化

**报告内容：**
- Task 6.1: 表格
- Task 6.2: 表格 + 图表

### Task 07: VAE and GAN
**输出文件：**
- `Task7_1_results.csv` - MSE和Inception Score对比表格
- `Task7_2_generated_images.png` - 生成图像对比

**报告内容：**
- Task 7.1: 表格
- Task 7.2: 图表（附录）

### Task 08: Reinforcement Learning
**输出文件：**
- `Task8_1_rewards.png` - 平均奖励曲线
- `Task8_1_final_performance.png` - 最终性能对比

**报告内容：**
- Task 8.1: 图表（Q-learning和SARSA对比）

### Task 09: Diffusion Model
**输出文件：**
- `Task9_1_noise_process.png` - 加噪过程可视化
- `Task9_2_training_loss.png` - 训练损失曲线

**报告内容：**
- Task 9.1: 图表（干净图像、加噪图像、噪声）
- Task 9.2: 代码 + 训练损失图表

### Task 10: YOLO Classification
**输出文件：**
- `{CID}_best.pt` - 最佳模型
- `{CID}_data.zip` - 标注数据集
- 学习曲线（YOLO自动生成）

**报告内容：**
- Task 10.1: 提交模型和数据集
- 报告中包含分类性能、训练参数、学习曲线

## 注意事项

1. **Task 04**: 脚本使用CIFAR-10作为演示数据集。如需使用Tiny-ImageNet，需要下载数据集并修改数据加载部分。

2. **Task 10**: 需要先收集和标注屋顶图像数据，组织成指定的目录结构后再运行脚本。

3. **训练时间**: 某些任务（特别是Task 04和Task 08）可能需要较长的训练时间。可以根据需要调整epochs参数。

4. **GPU加速**: 所有脚本都支持GPU加速，会自动检测并使用可用的GPU。

5. **中文支持**: 所有图表都使用中文标签，如果显示有问题，请确保系统安装了中文字体。

## 依赖库

确保安装以下Python库：
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn gymnasium ultralytics
```

## 下一步

1. 运行所有脚本生成结果
2. 收集所有输出的图表和表格
3. 根据报告要求将结果整理到LaTeX报告中
4. 对于Task 10，需要额外收集数据并训练YOLO模型

祝你顺利完成课程作业！
