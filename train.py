import torch
import deepinv
from torchvision import datasets, transforms
import tqdm
import time
import os
# 导入配置文件
from config import (
    device, dataset_name, large_model, batch_size, image_size, img_channels,
    num_classes, lr, epochs, timestep, get_output_model_path, get_transform,
    create_diffusion_params
)

# 设置matplotlib字体以支持中文显示
import matplotlib.pyplot as plt

# 打印设备信息
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# 获取输出模型名称
output_model_name = get_output_model_path()

# 获取数据变换
transform = get_transform()

# 创建扩散参数
diffusion_params = create_diffusion_params()
betas = diffusion_params['betas']
alphas = diffusion_params['alphas']
alphas_cumprod = diffusion_params['alphas_cumprod']
sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']



plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


if dataset_name=='CIFAR100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
if dataset_name=='CIFAR10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
if dataset_name=='MNIST':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)


from model import DiffUNet

import os

# 初始化模型
model = DiffUNet(in_channels=img_channels, out_channels=img_channels, num_classes=num_classes,large_model=large_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss()

# 用于存储训练历史
train_loss_history = []

# 检查是否存在已保存的模型
if os.path.exists(output_model_name):
    try:
        # 加载已保存的模型权重
        checkpoint = torch.load(output_model_name)
        model.load_state_dict(checkpoint)
        print(f"✅ 成功加载已保存的模型 '{output_model_name}'，将继续训练")
    except Exception as e:
        print(f"⚠️ 加载模型时出错: {e}")
        print("🔄 开始新的训练")
else:
    print(f"🔍 未找到已保存的模型 '{output_model_name}'，开始新的训练")

# 开始训练
total_start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_start_time = time.time()

    # 创建批次级进度条
    batch_progress = tqdm.tqdm(enumerate(train_loader),
                               total=len(train_loader),
                               desc=f"Epoch {epoch + 1}/{epochs}",
                               leave=False,
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    for batch_idx, (data, labels) in batch_progress:  # 注意这里现在接收data和labels
        # 计算批次开始时间
        batch_start_time = time.time()

        # 准备数据和类别标签
        imgs = data.to(device)
        labels = labels.to(device)  # 将类别标签移至设备
        noise = torch.randn_like(imgs)
        t = torch.randint(0, timestep, (imgs.shape[0],), device=device)

        # 前向扩散过程
        noisy_imgs = sqrt_alphas_cumprod[t, None, None, None] * imgs + \
                     sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise

        # 模型预测与损失计算 - 传递类别标签y
        optimizer.zero_grad()
        # 关键修改：将类别标签labels作为y参数传递给模型
        estimated_noise = model(noisy_imgs, t, y=labels, type_t="timestep")
        loss = mse(estimated_noise, noise)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 累计损失
        epoch_loss += loss.item()

        # 计算批次处理时间和每秒样本数
        batch_time = time.time() - batch_start_time
        samples_per_second = imgs.shape[0] / batch_time if batch_time > 0 else 0

        # 更新进度条信息
        current_lr = optimizer.param_groups[0]['lr']
        avg_batch_loss = epoch_loss / (batch_idx + 1)

        # 准备进度条后缀信息
        postfix_info = {
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{avg_batch_loss:.6f}',
            'lr': f'{current_lr:.6f}',
            'samples/s': f'{samples_per_second:.1f}'
        }

        # 如果在GPU上训练，添加GPU内存使用信息
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1024 ** 2  # MB
            postfix_info['gpu_mem'] = f'{gpu_memory:.1f}MB'

        batch_progress.set_postfix(postfix_info)

    # 计算本轮epoch的平均损失和时间
    avg_epoch_loss = epoch_loss / len(train_loader)
    epoch_time = time.time() - epoch_start_time
    train_loss_history.append(avg_epoch_loss)

    # 计算预计剩余时间
    elapsed_time = time.time() - total_start_time
    avg_epoch_time = elapsed_time / (epoch + 1)
    remaining_time = avg_epoch_time * (epochs - epoch - 1)

    # 打印本轮epoch的详细信息
    print(f"Epoch {epoch + 1}/{epochs} - 平均损失: {avg_epoch_loss:.6f} - 用时: {epoch_time:.2f}秒 - "
          f"预计剩余: {remaining_time // 60:.0f}分{remaining_time % 60:.0f}秒")

    # 保存模型
    torch.save(model.state_dict(), output_model_name)
print(f"训练完成！总用时: {time.time() - total_start_time:.2f}秒")
print(f"模型已保存为 '{output_model_name}'")

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_loss_history, '-b', label='训练损失')
plt.title('训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.grid(True)
plt.legend()
plt.tight_layout()
# 创建保存图像的文件夹
os.makedirs('./pic/train', exist_ok=True)
plt.savefig(f'./pic//train/{dataset_name}_{"large" if large_model else "small"}_diffusion_loss.png')
plt.close()
print(f"✅ 训练损失结果已保存到 './pic/train/{dataset_name}_{"large" if large_model else "small"}_diffusion_loss.png'")



