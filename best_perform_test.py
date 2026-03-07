import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import imageio
import glob


# --- 1. 增强版NCA模型 ---
class EnhancedStableNCA(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128):
        super().__init__()
        self.channel_n = channel_n

        # 更深的网络
        self.fc1 = nn.Conv2d(channel_n * 4, hidden_n, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden_n, hidden_n, kernel_size=1)  # 增加中间层
        self.fc3 = nn.Conv2d(hidden_n, channel_n, kernel_size=1)

        # 更好的初始化
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc3.bias)

    def perceive(self, x):
        # 梯度感知
        def gradient(x):
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0
            grad_x = F.conv2d(x, kernel_x.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1),
                              groups=self.channel_n, padding=1)
            grad_y = F.conv2d(x, kernel_y.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1),
                              groups=self.channel_n, padding=1)
            return grad_x, grad_y

        # 拉普拉斯算子
        def laplace(x):
            kernel = torch.tensor([[0.05, 0.2, 0.05],
                                   [0.2, -1.0, 0.2],
                                   [0.05, 0.2, 0.05]], dtype=torch.float32)
            return F.conv2d(x, kernel.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1),
                            groups=self.channel_n, padding=1)

        grad_x, grad_y = gradient(x)
        lap = laplace(x)

        return torch.cat([x, grad_x, grad_y, lap], dim=1)

    def get_alive_mask(self, x):
        # 使用多个通道判断存活
        alpha = x[:, 3:4, :, :]
        alive = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.01
        return alive

    def forward(self, x, steps=1, update_rate=0.5):
        for _ in range(steps):
            alive_mask = self.get_alive_mask(x)

            # 感知并更新
            obs = self.perceive(x)
            h = F.relu(self.fc1(obs))
            h = F.relu(self.fc2(h))
            update = self.fc3(h) * 0.05  # 更小的更新步长

            # 随机更新掩码
            rand_mask = (torch.rand(*x.shape[2:], device=x.device) < update_rate).float()

            # 更新（添加残差连接）
            x = x + update * rand_mask.view(1, 1, *x.shape[2:])

            # 应用存活掩码
            x = x * alive_mask.float()

            # 确保种子点保持活跃
            x[:, 3:, 32, 32] = 1.0

            # 数值稳定
            x = torch.tanh(x)

        return x


# --- 2. 改进的目标图像生成 ---
def get_target_circle_improved(size=64, radius=20):
    """生成更平滑的圆形"""
    target = np.zeros((size, size, 4), dtype=np.float32)
    center = size // 2

    # 使用向量化操作提高效率
    y, x = np.ogrid[-center:size - center, -center:size - center]
    distance = np.sqrt(x ** 2 + y ** 2)

    # 创建平滑的边缘
    mask = distance <= radius
    target[mask] = [0.2, 0.6, 1.0, 1.0]

    return torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)


# --- 3. 创建生长动画函数 ---
def create_growth_animation(model, steps=200, save_dir="growth_animation", device="cpu", fps=15):
    """创建并保存生长动画"""
    os.makedirs(save_dir, exist_ok=True)

    # 初始化状态
    x = torch.zeros(1, 16, 64, 64).to(device)
    x[:, 3:, 32, 32] = 1.0  # 中心种子

    frames = []

    print(f"生成生长动画 ({steps}步)...")
    start_time = datetime.now()

    for s in range(steps):
        x = model(x, steps=1, update_rate=0.5)

        if s % 2 == 0:  # 每2步保存一帧
            # 提取图像
            img = x[0, :3].detach().cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # 从[-1,1]映射到[0,1]
            img = np.clip(img, 0, 1)

            # 保存为PNG
            frame_path = os.path.join(save_dir, f"frame_{s:04d}.png")
            plt.imsave(frame_path, img)
            frames.append(img)

            # 显示进度
            if s % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"  步数: {s}/{steps} | 已用时间: {elapsed:.1f}秒")

    # 创建GIF
    gif_path = os.path.join(save_dir, "growth_animation.gif")
    if frames:
        # 将帧转换为uint8
        frames_uint8 = [(frame * 255).astype(np.uint8) for frame in frames]
        imageio.mimsave(gif_path, frames_uint8, fps=fps)
        print(f"GIF动画已保存: {gif_path}")

    # 保存最终图像
    final_img = x[0, :3].detach().cpu().permute(1, 2, 0).numpy()
    final_img = (final_img + 1) / 2
    final_path = os.path.join(save_dir, "final_result.png")
    plt.imsave(final_path, final_img)
    print(f"最终图像已保存: {final_path}")

    elapsed_total = (datetime.now() - start_time).total_seconds()
    print(f"动画生成完成！总耗时: {elapsed_total:.1f}秒")

    return save_dir


# --- 4. 简化的训练函数（修复verbose问题）---
def train_nca_simple(model, target_img, device, epochs=2000):
    """简化的训练函数，避免版本兼容性问题"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 初始化池
    POOL_SIZE = 32
    seed = torch.zeros(1, 16, 64, 64)
    seed[:, 3:, 32, 32] = 1.0
    pool = seed.clone().repeat(POOL_SIZE, 1, 1, 1).to(device)

    print("开始训练...")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    loss_history = []
    best_loss = float('inf')

    for epoch in range(epochs):
        # 采样
        batch_size = 4
        batch_idx = np.random.choice(POOL_SIZE, batch_size, replace=False)
        x = pool[batch_idx]

        # 定期重置
        if epoch % 50 == 0:
            x[0] = seed.to(device)

        # 演化
        steps = np.random.randint(40, 80)
        out = model(x, steps=steps, update_rate=0.5)

        # 计算损失
        loss = F.mse_loss(out[:, :4], target_img.repeat(batch_size, 1, 1, 1))

        # 优化
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 更新池
        pool[batch_idx] = out.detach()

        # 记录损失
        loss_history.append(loss.item())

        # 保存最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'loss_history': loss_history,
            }, 'best_nca_model.pth')
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: 保存最佳模型，损失: {best_loss:.6f}")

        # 打印进度
        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d}/{epochs} | Loss: {loss.item():.6f} | Best: {best_loss:.6f}")

            # 定期可视化
            if epoch % 500 == 0:
                visualize_training_progress(model, seed.to(device), target_img,
                                            loss_history, epoch, device)

        # 定期重新初始化池
        if epoch % 1000 == 0 and epoch > 0:
            pool = seed.clone().repeat(POOL_SIZE, 1, 1, 1).to(device)
            print(f"Epoch {epoch}: 重置样本池")

    print(f"训练完成！最佳损失: {best_loss:.6f}")
    return model, loss_history


# --- 5. 可视化训练进度 ---
def visualize_training_progress(model, seed, target_img, loss_history, epoch, device):
    """可视化训练进度"""
    with torch.no_grad():
        # 生成当前模型的结果
        out = model(seed, steps=100, update_rate=0.5)
        img = out[0, :3].detach().cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        # 目标图像
        target_np = target_img[0].permute(1, 2, 0).cpu().numpy()
        target_np = np.clip(target_np[:, :, :3], 0, 1)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 图像对比
        axes[0].imshow(target_np)
        axes[0].set_title("Target Image")
        axes[0].axis('off')

        axes[1].imshow(img)
        axes[1].set_title(f"Generated (Epoch {epoch})")
        axes[1].axis('off')

        # 损失曲线
        axes[2].plot(loss_history)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f"NCA Training Progress - Epoch {epoch}", fontsize=14)
        plt.tight_layout()

        # 保存图像
        progress_path = f'training_progress_epoch_{epoch:05d}.png'
        plt.savefig(progress_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"训练进度图已保存: {progress_path}")


# --- 6. 测试不同种子 ---
def test_different_seeds(model, steps=100, device="cpu"):
    """测试不同的初始种子"""
    seeds_config = [
        ("Center", lambda: torch.zeros(1, 16, 64, 64).to(device)),
        ("Four Corners", lambda: torch.zeros(1, 16, 64, 64).to(device)),
    ]

    # 设置四角种子
    four_corners = torch.zeros(1, 16, 64, 64).to(device)
    four_corners[:, 3:, 16, 16] = 1.0  # 左上
    four_corners[:, 3:, 16, 48] = 1.0  # 右上
    four_corners[:, 3:, 48, 16] = 1.0  # 左下
    four_corners[:, 3:, 48, 48] = 1.0  # 右下

    # 替换lambda函数
    seeds_config[1] = ("Four Corners", lambda: four_corners.clone())

    results = []

    for name, seed_func in seeds_config:
        x = seed_func()

        # 生长
        for _ in range(steps):
            x = model(x, steps=1, update_rate=0.5)

        # 提取结果
        img = x[0, :3].detach().cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        results.append((name, img))

    # 显示结果
    plt.figure(figsize=(8, 4))
    for idx, (name, img) in enumerate(results):
        plt.subplot(1, 2, idx + 1)
        plt.imshow(img)
        plt.title(f"Seed: {name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return results


# --- 7. 快速演示函数 ---
def quick_demo(model, steps=100, device="cpu"):
    """快速演示模型效果"""
    print("快速演示模型效果...")

    # 从中心种子开始
    seed = torch.zeros(1, 16, 64, 64).to(device)
    seed[:, 3:, 32, 32] = 1.0

    x = seed.clone()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 不同步数的演示
    demo_steps = [0, 10, 25, 50, 75, 100]

    for idx, step in enumerate(demo_steps):
        if step > 0:
            x = model(x, steps=step, update_rate=0.5)

        img = x[0, :3].detach().cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        ax = axes[idx // 3, idx % 3]
        ax.imshow(img)
        ax.set_title(f"Step {step}")
        ax.axis('off')

    plt.suptitle("NCA Growth Demo", fontsize=16)
    plt.tight_layout()
    plt.show()


# --- 8. 检查并创建目录 ---
def ensure_directory(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        print(f"创建目录: {path}")
        os.makedirs(path, exist_ok=True)
    return path


# --- 9. 主程序 ---
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")

    # 创建模型
    model = EnhancedStableNCA(channel_n=16, hidden_n=128).to(device)

    # 生成目标图像
    target_img = get_target_circle_improved().to(device)

    # 显示目标
    plt.figure(figsize=(6, 6))
    target_np = target_img[0].permute(1, 2, 0).cpu().numpy()
    plt.imshow(np.clip(target_np[:, :, :3], 0, 1))
    plt.title("Target: Blue Circle")
    plt.axis('off')
    plt.show()

    # 确保G:/NA目录存在
    base_path = "G:/NA"
    ensure_directory(base_path)

    # 询问是否训练
    print("\n" + "=" * 50)
    print("NCA训练和演示程序")
    print("=" * 50)

    train_choice = input("\n是否开始训练？(y/n，建议选y): ").lower().strip()

    if train_choice == 'y':
        # 训练模型
        epochs_input = input(f"输入训练轮次 (默认1000，建议2000以上效果更好): ").strip()
        epochs = int(epochs_input) if epochs_input else 1000

        print(f"\n开始训练，轮次: {epochs}")
        print("训练过程中会定期保存最佳模型...")

        model, loss_history = train_nca_simple(model, target_img, device, epochs=epochs)

        # 保存最终结果
        save_path = ensure_directory(os.path.join(base_path, "nca_results"))

        # 保存最终模型
        final_model_path = os.path.join(save_path, "final_nca_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'target_img': target_img.cpu(),
            'epochs': epochs,
        }, final_model_path)
        print(f"\n最终模型已保存: {final_model_path}")

        # 绘制损失曲线
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss History (Best: {min(loss_history):.6f})')
        plt.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(save_path, "training_loss.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"损失曲线已保存: {loss_plot_path}")

    else:
        # 加载预训练模型（如果存在）
        save_path = ensure_directory(os.path.join(base_path, "nca_results"))
        model_path = os.path.join(save_path, "final_nca_model.pth")

        if os.path.exists(model_path):
            print("加载预训练模型...")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            loss_history = checkpoint.get('loss_history', [])
            print(f"模型加载成功，训练轮次: {checkpoint.get('epochs', '未知')}")
            print(f"历史损失记录: {len(loss_history)}个")
        else:
            print("未找到预训练模型，使用随机初始化的模型")
            print("注意：随机模型可能无法生成好的结果，建议先训练")
            loss_history = []

    # 创建生长动画
    print("\n" + "=" * 50)
    print("创建生长动画")
    print("=" * 50)

    animation_steps = input("输入动画步数 (默认150): ").strip()
    steps = int(animation_steps) if animation_steps else 150

    # 创建带时间戳的动画目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    animation_dir = os.path.join(save_path, f"animation_{timestamp}")

    print(f"\n开始创建动画，步数: {steps}")
    animation_result = create_growth_animation(
        model,
        steps=steps,
        save_dir=animation_dir,
        device=device,
        fps=10  # 降低帧率使动画更平滑
    )

    # 快速演示
    print("\n" + "=" * 50)
    print("快速演示")
    print("=" * 50)

    demo_choice = input("是否显示快速演示？(y/n): ").lower().strip()
    if demo_choice == 'y':
        quick_demo(model, steps=100, device=device)

    # 测试不同种子
    print("\n" + "=" * 50)
    print("测试不同初始种子")
    print("=" * 50)

    seed_test_choice = input("是否测试不同种子？(y/n): ").lower().strip()
    if seed_test_choice == 'y':
        test_different_seeds(model, steps=80, device=device)

    print(f"\n" + "=" * 50)
    print("程序完成！")
    print(f"所有结果已保存到: {save_path}")
    print(f"动画文件在: {animation_dir}")
    print("=" * 50)