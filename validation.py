import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import random
import os
# 导入配置文件
from config import (
    device, dataset_name, large_model, image_size, img_channels,
    num_classes, batch_size, timestep, get_input_model_path, get_transform,
    create_diffusion_params, validation_n_samples
)
from model import DiffUNet  # 模型定义

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 打印设备信息
print(f"使用设备: {device}")

# 获取输入模型名称（验证时使用）
input_model_name = get_input_model_path()

# 获取数据变换
transform = get_transform()

# 创建扩散参数
diffusion_params = create_diffusion_params()
betas = diffusion_params['betas']
alphas = diffusion_params['alphas']
alphas_cumprod = diffusion_params['alphas_cumprod']
sqrt_recip_alphas = diffusion_params['sqrt_recip_alphas']
sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']

if dataset_name=='CIFAR100':
    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)
if dataset_name=='CIFAR10':
    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)
if dataset_name=='MNIST':
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)




# 模型初始化并加载权重
model = DiffUNet(in_channels=img_channels, out_channels=img_channels, num_classes=num_classes,large_model=large_model).to(device)
model.load_state_dict(torch.load(input_model_name, map_location=device))
model.eval()
print("✅ 模型已加载")



@torch.no_grad()
def p_sample(model, x, t, y):
    """反向一步采样"""
    # 把 t 转成 batch 张量
    t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

    betas_t = betas[t].reshape(1, 1, 1, 1)
    sqrt_recip_alpha_t = sqrt_recip_alphas[t].reshape(1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(1, 1, 1, 1)

    # 预测噪声
    model_mean = sqrt_recip_alpha_t * (
        x - betas_t / sqrt_one_minus_alphas_cumprod_t * model(x, t_batch, y=y, type_t="timestep")
    )

    if t > 0:   # 现在 t 是 int，不会再报错
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise
    else:
        return model_mean


@torch.no_grad()
def p_sample_loop(model, y, shape):
    """完整反向扩散过程"""
    device = next(model.parameters()).device
    img = torch.randn(shape, device=device)
    for t in reversed(range(timestep)):
        img = p_sample(model, img, t, y)  # 直接传 int t
    return img
@torch.no_grad()
def fast_p_sample_loop(model, y, shape, num_steps=50):
    """
    快速反向扩散采样 (子采样版本)
    - num_steps: 实际采样步数 (比如 50)
    """
    device = next(model.parameters()).device
    img = torch.randn(shape, device=device)

    # 选取 num_steps 个子采样点（等间隔）
    step_indices = torch.linspace(0, timestep - 1, num_steps, dtype=torch.long)

    for i in reversed(range(num_steps)):
        t = step_indices[i].item()  # 当前简化的时间步 (int)
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # --- 关键点：直接用与原始 p_sample 相同的逻辑 ---
        betas_t = betas[t].reshape(1, 1, 1, 1)
        sqrt_recip_alpha_t = sqrt_recip_alphas[t].reshape(1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(1, 1, 1, 1)

        # 模型预测噪声
        model_mean = sqrt_recip_alpha_t * (
            img - betas_t / sqrt_one_minus_alphas_cumprod_t * model(img, t_batch, y=y, type_t="timestep")
        )

        if i > 0:  # 不是最后一步就加噪声
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(betas_t) * noise
        else:
            img = model_mean

    return img
# ========== 验证与可视化 ==========
data_iter = iter(test_loader)
imgs, labels = next(data_iter)
imgs, labels = imgs.to(device), labels.to(device)

# 从测试集中随机挑选几个类别生成
n_samples = validation_n_samples
idx = random.sample(range(imgs.size(0)), n_samples)
sample_imgs = imgs[idx]
sample_labels = labels[idx]

# 使用模型生成图片
generated_imgs = p_sample_loop(model, sample_labels, sample_imgs.shape)#原始版本

# 反归一化到 [0,1] 便于显示
def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2  # 假设训练时标准化到 [-1,1]，否则改成 [0,1]

orig = denorm(sample_imgs).cpu().permute(0, 2, 3, 1).numpy()
gen = denorm(generated_imgs).cpu().permute(0, 2, 3, 1).numpy()

# 可视化
fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))

for i in range(n_samples):
    axes[0, i].imshow(orig[i])
    axes[0, i].set_title(f"原始 (类 {sample_labels[i].item()})")
    axes[0, i].axis("off")

    axes[1, i].imshow(gen[i])
    axes[1, i].set_title("生成")
    axes[1, i].axis("off")

plt.tight_layout()

# 创建保存图像的文件夹
os.makedirs('./pic/val', exist_ok=True)
plt.savefig(f'./pic/val/{dataset_name}_{"large" if large_model else "small"}_diffusion.png')
plt.close()
print(f"✅ 生成结果已保存到 './pic/val/{dataset_name}_{"large" if large_model else "small"}_diffusion.png'")
