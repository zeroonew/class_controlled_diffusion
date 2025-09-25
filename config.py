import torch

# ====== 设备配置 ======
# 自动选择可用设备
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# ====== 数据集配置 ======
dataset_name = 'MNIST'  # 可选: 'CIFAR100', 'CIFAR10', 'MNIST'
if dataset_name=='MNIST':
    image_size = 28  # 图像尺寸
    img_channels = 1  # 图像通道数 (MNIST)
    num_classes = 10  # 类别数量 (MNIST)
elif dataset_name=='CIFAR10':
    image_size = 32  # 图像尺寸
    img_channels = 3  # 图像通道数 (CIFAR10)
    num_classes = 10  # 类别数量 (CIFAR10)
elif dataset_name=='CIFAR100':
    image_size = 32  # 图像尺寸
    img_channels = 3  # 图像通道数 (CIFAR100)
    num_classes = 100  # 类别数量 (CIFAR100)

# ====== 模型配置 ======
large_model = False  # 是否使用大模型

# ====== 训练配置 ======
batch_size = 64  # 训练批次大小
epochs = 100  # 训练轮数
lr = 1e-4  # 学习率

# ====== 扩散模型参数 ======
beta_start = 1e-4  # 噪声调度起始值
beta_end = 0.02  # 噪声调度结束值
timestep = 1000  # 时间步数

# ====== 文件路径配置 ======
# 输出模型名称
def get_output_model_path():
    return f'./trained_models/{dataset_name}_{"large" if large_model else "small"}_diffusion_model.pth'

# 输入模型名称 (验证时使用)
def get_input_model_path():
    return get_output_model_path()

# ====== 数据变换配置 ======
from torchvision import transforms

def get_transform():
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1,))
    ])

# ====== 生成扩散参数 ======
# 创建噪声调度参数
def create_diffusion_params(device=device):
    betas = torch.linspace(beta_start, beta_end, timestep, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas
    }

# ====== 可视化配置 ======
validation_n_samples = 8  # 验证时生成的样本数量