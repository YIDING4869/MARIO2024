import torch

from config import config
# 打印PyTorch版本
print("PyTorch Version:", torch.__version__)

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# 如果CUDA可用，打印CUDA设备名称
if cuda_available:
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i} Name:", torch.cuda.get_device_name(i))
else:
    print("No CUDA Device Found, using CPU")

# 测试在CUDA设备上创建张量
if cuda_available:
    x = torch.randn(3, 3).cuda()
    print("Tensor on CUDA Device:", x)
else:
    x = torch.randn(3, 3)
    print("Tensor on CPU:", x)

print(config.gamma)
