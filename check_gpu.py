import torch

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda if torch.cuda.is_available() else "未安装")
print("可用GPU数量:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

if torch.cuda.is_available():
    print("GPU名称:", torch.cuda.get_device_name(0))
else:
    print("\n原因可能是:")
    print("1. 系统没有NVIDIA GPU")
    print("2. 安装的是CPU版本的PyTorch")
    print("3. CUDA驱动未正确安装")
    print("\n如需使用GPU，请安装GPU版本的PyTorch:")
    print("访问 https://pytorch.org 选择对应的CUDA版本安装")
