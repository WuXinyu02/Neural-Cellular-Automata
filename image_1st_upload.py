import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import imageio
import glob
from PIL import Image
import time
from tqdm import tqdm  # 需要安装：pip install tqdm
from typing import Tuple, List
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda.amp as amp  # 混合精度训练

# --- 0.设置中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("中文字体设置成功")
    except:
        print("使用默认字体（中文可能显示为方块）")

# 在主程序开头调用
setup_chinese_font()


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


# --- 2. 从图像文件加载目标图像（保留原始尺寸）---
def load_target_from_image(image_path, target_size=None, device=None):
    """
    从图像文件加载并处理目标图像

    参数:
    - image_path: 图像文件路径
    - target_size: 目标尺寸，如果为None则使用原始尺寸
    - device: 设备 (CPU/GPU)

    返回:
    - target_tensor: 形状为 (1, 4, H, W) 的RGBA目标张量
    - original_rgb: 原始RGB图像（用于显示）
    - original_size: 原始图像尺寸 (height, width)
    """
    print(f"加载目标图像: {image_path}")

    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 加载图像
    img = Image.open(image_path).convert('RGBA')
    original_size = img.size[::-1]  # PIL返回(width, height)，转换为(height, width)

    # 如果指定了目标尺寸，则调整大小；否则使用原始尺寸
    if target_size is not None:
        img = img.resize(target_size[::-1] if target_size else original_size,
                         Image.Resampling.LANCZOS)
        target_size = target_size
    else:
        target_size = original_size

    # 转换为numpy数组 [0-255] 并归一化到 [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0

    # 提取RGBA通道
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    a = img_array[:, :, 3]

    # 创建目标张量 (4通道 RGBA)
    target = np.zeros((4, target_size[0], target_size[1]), dtype=np.float32)
    target[0] = r * 2.0 - 1.0  # 映射到[-1, 1]
    target[1] = g * 2.0 - 1.0
    target[2] = b * 2.0 - 1.0
    target[3] = a * 2.0 - 1.0

    # 转换为torch张量
    target_tensor = torch.from_numpy(target).unsqueeze(0)

    if device:
        target_tensor = target_tensor.to(device)

    # 保存原始RGB图像用于显示
    original_rgb = img_array[:, :, :3]

    print(f"目标图像加载完成，原始尺寸: {original_size}, 处理尺寸: {target_size}, 通道数: 4")

    return target_tensor, original_rgb, original_size


# --- 3. 创建生长动画函数（支持动态尺寸）---
def create_growth_animation(model, steps=200, save_dir="growth_animation", device="cpu",
                           fps=15, image_size=(64, 64)):
    """创建并保存生长动画，支持自定义图像尺寸"""
    os.makedirs(save_dir, exist_ok=True)

    # 初始化状态 - 使用指定的图像尺寸
    height, width = image_size
    x = torch.zeros(1, 16, height, width).to(device)
    x[:, 3:, height//2, width//2] = 1.0  # 中心种子

    frames = []

    print(f"生成生长动画 ({steps}步, 尺寸: {width}x{height})...")
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


# --- 4. 改进的训练函数（支持动态尺寸）---



def train_nca_with_timing(model, target_img, device, epochs=2000, save_dir="G:/NA/nca_results",
                          image_size=(64, 64)) -> Tuple[nn.Module, List[float]]:
    # 初始化返回值
    loss_history: List[float] = []

    # 计算所需内存
    height, width = image_size
    estimated_memory = height * width * 16 * 4 * 32 * 4 / (1024 ** 3)

    print(f"估计所需显存: {estimated_memory:.2f} GB")

    # 内存检查
    if estimated_memory > 8:
        print("⚠️ 警告：图像尺寸过大，可能导致内存不足！")
        print(f"当前尺寸: {width}x{height}")
        print("建议尺寸: ≤512x512")
        print("\n选项：")
        print("1. 继续训练（可能失败）")
        print("2. 取消训练，返回主菜单")

        continue_choice = input("选择 (1-2，默认2): ").strip()
        if continue_choice != '1':
            print("已取消训练")
            return model, loss_history

    # 创建进度保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 优化1: 启用cudnn自动优化
    torch.backends.cudnn.benchmark = True

    # 优化2: 动态调整参数
    if height * width > 256 * 256:
        batch_size = 2
        POOL_SIZE = 16
        print(f"检测到大尺寸图像，已调整batch_size={batch_size}, pool_size={POOL_SIZE}")
    else:
        batch_size = 4
        POOL_SIZE = 32

    # 初始化池
    seed = torch.zeros(1, 16, height, width)
    seed[:, 3:, height // 2, width // 2] = 1.0
    pool = seed.clone().repeat(POOL_SIZE, 1, 1, 1).to(device)

    # 优化3: 创建预先生成的索引池，避免每次随机选择
    import itertools
    indices_cycle = itertools.cycle(range(POOL_SIZE))

    print("=" * 60)
    print("开始训练 NCA 模型")
    print("=" * 60)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练设备: {device}")
    print(f"目标轮次: {epochs}")
    print(f"图像尺寸: {width}x{height}")
    print(f"Batch Size: {batch_size}")
    print(f"Pool Size: {POOL_SIZE}")
    print(f"保存目录: {save_dir}")
    print("=" * 60)

    # 优化4: 预编译计算图（如果使用较新PyTorch）
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("使用torch.compile优化模型...")
        model = torch.compile(model, mode="reduce-overhead")
    elif hasattr(torch, 'compile') and device.type == 'cpu':
        print("ℹ️ CPU模式下跳过torch.compile（需要C++编译器才能使用）")
    # 优化5: 为GPU传输创建固定内存的池
    if device.type == 'cuda':
        pool = pool.pin_memory()  # 加速CPU到GPU的传输

    # 初始化计时器
    start_time = time.time()
    best_loss = float('inf')
    epoch_times = []

    # 优化6: 创建CUDA流实现异步传输
    if device.type == 'cuda':
        stream = torch.cuda.Stream()
    else:
        stream = None

    # 训练循环
    for epoch in range(epochs):
        try:
            # 记录本轮开始时间
            epoch_start = time.time()

            # 优化7: 批量采样 - 使用预先生成的索引
            batch_idx = [next(indices_cycle) for _ in range(batch_size)]
            x = pool[batch_idx]

            # 优化8: 异步数据传输
            if stream is not None:
                with torch.cuda.stream(stream):
                    x = x.to(device, non_blocking=True)
            else:
                x = x.to(device)

            # 定期重置
            if epoch % 50 == 0:
                seed_gpu = seed.to(device, non_blocking=True)
                x[0] = seed_gpu

            # 演化
            steps = np.random.randint(40, 80)
            out = model(x, steps=steps, update_rate=0.5)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # 计算损失
            target_repeated = target_img.repeat(batch_size, 1, 1, 1)
            loss = F.mse_loss(out[:, :4], target_repeated)

            # 优化
            optimizer.zero_grad(set_to_none=True)  # 比zero_grad()更快
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 优化9: 异步更新池
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)  # 等待传输完成

            # 更新池
            pool[batch_idx] = out.detach().cpu()

            # 记录损失
            loss_history.append(loss.item())

            # 计算本轮耗时
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # 保存最佳模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'loss_history': loss_history,
                }, os.path.join(save_dir, 'best_nca_model.pth'))

                if epoch % 100 == 0:
                    print(f"✓ Epoch {epoch:5d}: 保存最佳模型，损失: {best_loss:.6f}")

            # 定期打印进度
            if epoch % 50 == 0:
                elapsed_total = time.time() - start_time
                avg_epoch_time = np.mean(epoch_times[-50:]) if epoch_times else 0

                print(f"\n📊 Epoch {epoch:5d}/{epochs} | Loss: {loss.item():.6f} | Best: {best_loss:.6f}")
                print(f"   ├─ 本轮耗时: {epoch_time:.2f}秒")
                print(f"   ├─ 已用时间: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")

                if avg_epoch_time > 0:
                    remaining = (epochs - epoch - 1) * avg_epoch_time
                    print(f"   └─ 预计剩余: {time.strftime('%H:%M:%S', time.gmtime(remaining))}")

                # 显示GPU利用率（如果有）
                if device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
                    gpu_cached = torch.cuda.memory_reserved() / 1024 ** 3
                    print(f"   GPU内存: 已用 {gpu_memory:.2f}GB / 缓存 {gpu_cached:.2f}GB")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ GPU内存不足！错误: {e}")
                print("建议减小图像尺寸或batch_size")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    # 训练完成总结
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)
    if loss_history:
        print(f"📈 最终统计:")
        print(f"   ├─ 总训练轮次: {len(loss_history)}")
        print(f"   ├─ 总训练时间: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"   ├─ 最终损失: {loss_history[-1]:.6f}")
        print(f"   ├─ 最佳损失: {best_loss:.6f}")
        if loss_history:
            improvement = (loss_history[0] - best_loss) / loss_history[0] * 100
            print(f"   └─ 损失改善: {improvement:.1f}%")
    else:
        print("⚠️ 训练未执行或提前终止")
    print("=" * 60)

    return model, loss_history


# --- 5. 改进的可视化训练进度函数（带保存路径）---
def visualize_training_progress(model, seed, target_img, loss_history, epoch, device, save_dir):
    """可视化训练进度"""
    with torch.no_grad():
        # 生成当前模型的结果
        gen_start = time.time()
        out = model(seed, steps=100, update_rate=0.5)
        gen_time = time.time() - gen_start

        img = out[0, :3].detach().cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        # 目标图像
        target_np = target_img[0].permute(1, 2, 0).cpu().numpy()
        target_np = np.clip(target_np[:, :, :3], 0, 1)

        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 图像对比
        axes[0, 0].imshow(target_np)
        axes[0, 0].set_title("Target Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img)
        axes[0, 1].set_title(f"Generated (Epoch {epoch})")
        axes[0, 1].axis('off')

        # 损失曲线
        axes[0, 2].plot(loss_history)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].grid(True, alpha=0.3)

        # 添加损失统计
        min_loss = min(loss_history)
        current_loss = loss_history[-1]
        axes[0, 2].axhline(y=min_loss, color='r', linestyle='--', alpha=0.5, label=f'Best: {min_loss:.4f}')
        axes[0, 2].legend()

        # 差异图
        diff = np.abs(target_np - img).mean(axis=2)
        im_diff = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Difference Map (Mean: {diff.mean():.3f})')
        axes[1, 0].axis('off')
        plt.colorbar(im_diff, ax=axes[1, 0])

        # 生成图像的通道统计
        channels = out[0, :4].detach().cpu().numpy()
        axes[1, 1].hist(channels.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title('Channel Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')

        # 训练信息
        axes[1, 2].axis('off')
        info_text = f"""
训练信息:
━━━━━━━━━━━━━━━━━
轮次: {epoch}
当前损失: {current_loss:.6f}
最佳损失: {min_loss:.6f}
损失改善: {(loss_history[0] - min_loss) / loss_history[0] * 100:.1f}%

生成用时: {gen_time:.2f}秒
图像尺寸: 64x64
通道数: 16
        """
        axes[1, 2].text(0.1, 0.5, info_text, fontsize=10,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f"NCA Training Progress - Epoch {epoch}", fontsize=14)
        plt.tight_layout()

        # 保存图像
        progress_path = os.path.join(save_dir, f'training_progress_epoch_{epoch:05d}.png')
        plt.savefig(progress_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 训练进度图已保存: {progress_path}")


# --- 6. 测试不同种子函数（支持动态尺寸）---
def test_different_seeds(model, steps=100, device="cpu", image_size=(64, 64)):
    """测试不同的初始种子，支持自定义图像尺寸"""
    print("\n测试不同初始种子的生长效果...")

    height, width = image_size
    center_h, center_w = height // 2, width // 2

    # 创建不同种子配置
    seed_configs = [
        ("中心单点", (center_h, center_w)),
        ("四角", [(10, 10), (10, width - 10), (height - 10, 10), (height - 10, width - 10)]),
        ("水平线", [(center_h, y) for y in range(20, width - 20, 10)]),
        ("垂直线", [(x, center_w) for x in range(20, height - 20, 10)]),
        ("随机点", [(np.random.randint(20, height - 20),
                     np.random.randint(20, width - 20)) for _ in range(5)]),
        ("十字形", [(center_h, center_w), (center_h - 1, center_w),
                    (center_h + 1, center_w), (center_h, center_w - 1), (center_h, center_w + 1)])
    ]

    # ... 其余代码保持不变 ...


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

    # 询问目标图像路径
    print("\n" + "=" * 50)
    print("NCA训练和演示程序")
    print("=" * 50)

    # 获取目标图像路径
    default_path = r"G:\NA\target.png"
    print(f"\n请指定目标图像路径（支持格式：PNG、JPG、JPEG、BMP等）")
    print(f"默认路径: {default_path}")

    image_path = input("输入目标图像路径 (直接回车使用默认): ").strip()
    if not image_path:
        image_path = default_path

    # 询问是否保持原始尺寸
    print("\n选择图像处理方式：")
    print("1. 保持原始尺寸（输出与原图相同大小）")
    print("2. 指定输出尺寸")
    size_choice = input("选择 (1-2，默认1): ").strip()

    try:
        if size_choice == '2':
            custom_width = int(input("输入图像宽度: "))
            custom_height = int(input("输入图像高度: "))
            target_size = (custom_height, custom_width)
            target_img, original_rgb, original_size = load_target_from_image(
                image_path, target_size=target_size, device=device
            )
            image_size = target_size
            print(f"使用自定义尺寸: {custom_width}x{custom_height}")
        else:
            # 保持原始尺寸
            target_img, original_rgb, original_size = load_target_from_image(
                image_path, target_size=None, device=device
            )
            image_size = original_size
            print(f"保持原始尺寸: {original_size[1]}x{original_size[0]}")

        # 创建模型（需要根据图像尺寸调整？）
        # 注意：模型本身是卷积的，可以处理任意尺寸，不需要重新创建
        model = EnhancedStableNCA(channel_n=16, hidden_n=128).to(device)

        # 显示目标图像
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_rgb)
        plt.title(f"Original Image\n{original_size[1]}x{original_size[0]}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        target_display = (target_img[0, :3].cpu().numpy() + 1) / 2
        target_display = np.clip(target_display.transpose(1, 2, 0), 0, 1)
        plt.imshow(target_display)
        plt.title(f"Processed Target\n{image_size[1]}x{image_size[0]}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        # 显示尺寸信息
        plt.text(0.1, 0.5,
                 f"图像信息:\n\n原始: {original_size[1]}x{original_size[0]}\n处理: {image_size[1]}x{image_size[0]}\n通道: RGBA (4)\n范围: [-1, 1]",
                 fontsize=12, verticalalignment='center')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"加载目标图像失败: {e}")
        print("使用默认的64x64圆形目标作为替代")


        def get_target_circle_improved(size=64, radius=20):
            target = np.zeros((size, size, 4), dtype=np.float32)
            center = size // 2
            y, x = np.ogrid[-center:size - center, -center:size - center]
            distance = np.sqrt(x ** 2 + y ** 2)
            mask = distance <= radius
            target[mask] = [0.2, 0.6, 1.0, 1.0]
            return torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)


        target_img = get_target_circle_improved().to(device)
        original_rgb = None
        image_size = (64, 64)

    # 确保G:/NA目录存在
    base_path = "G:/NA"
    save_path = ensure_directory(os.path.join(base_path, "nca_results"))

    # 询问是否训练
    print("\n" + "=" * 50)
    train_choice = input("是否开始训练？(y/n，建议选y): ").lower().strip()

    if train_choice == 'y':
        # 训练模型
        epochs_input = input(f"输入训练轮次 (默认1000，建议2000以上效果更好): ").strip()
        epochs = int(epochs_input) if epochs_input else 1000

        print(f"\n开始训练，轮次: {epochs}")
        print(f"训练图像尺寸: {image_size[1]}x{image_size[0]}")
        print("训练过程中会定期保存最佳模型...")

        # 使用带计时功能的训练函数
        model, loss_history = train_nca_with_timing(
            model,
            target_img,
            device,
            epochs=epochs,
            save_dir=save_path,
            image_size=image_size  # 传入图像尺寸
        )

        # 保存最终模型
        final_model_path = os.path.join(save_path, "final_nca_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'target_img': target_img.cpu(),
            'target_path': image_path,
            'epochs': epochs,
            'image_size': image_size,
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
        model_path = os.path.join(save_path, "final_nca_model.pth")

        if os.path.exists(model_path):
            print("加载预训练模型...")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            loss_history = checkpoint.get('loss_history', [])
            image_size = checkpoint.get('image_size', (64, 64))
            print(f"模型加载成功，训练轮次: {checkpoint.get('epochs', '未知')}")
            print(f"模型图像尺寸: {image_size[1]}x{image_size[0]}")
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

    # 选择帧率
    print("\n选择帧率：")
    print("1. 15 fps (标准)")
    print("2. 24 fps (电影)")
    print("3. 30 fps (高清)")
    fps_choice = input("选择 (1-3，默认1): ").strip()

    fps_map = {'1': 15, '2': 24, '3': 30}
    fps = fps_map.get(fps_choice, 15)

    # 创建带时间戳的动画目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    animation_dir = os.path.join(save_path, f"animation_{timestamp}")

    print(f"\n开始创建动画，步数: {steps}, 帧率: {fps}fps, 尺寸: {image_size[1]}x{image_size[0]}")
    animation_result = create_growth_animation(
        model,
        steps=steps,
        save_dir=animation_dir,
        device=device,
        fps=fps,
        image_size=image_size  # 传入图像尺寸
    )

    # ... 其余代码保持不变 ...