# plot_loss.py —— 读取 train_loss.csv 绘制 loss 曲线，保存为 PNG
# 用法：python plot_loss.py [csv路径] [输出PNG路径]（训练过程中可随时重复运行）
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 无界面后端，直接输出图片文件
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_CSV = os.path.join("checkpoints", "train_loss.csv")
DEFAULT_PNG = os.path.join("checkpoints", "loss_curve.png")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    png_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PNG

    steps, losses = [], []
    try:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0].strip().isdigit():
                    steps.append(int(row[0]))
                    losses.append(float(row[1]))
    except FileNotFoundError:
        print(f"未找到 {csv_path}，请先运行训练产生数据")
        return 1
    if len(losses) < 2:
        print(f"数据点不足（{len(losses)} 个），至少需要 2 个点才能绘图")
        return 1

    # 移动平均平滑（窗口为总点数 1/50，最小 2）
    win = max(2, len(losses) // 50)
    smoothed = []
    for i in range(len(losses)):
        seg = losses[max(0, i - win + 1):i + 1]
        smoothed.append(sum(seg) / len(seg))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, color="#8888cc", lw=0.6, alpha=0.6, label="loss")
    ax.plot(steps, smoothed, color="#cc4444", lw=1.6, label="平滑")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"训练 loss 曲线（step {steps[0]} ~ {steps[-1]}）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"已保存: {png_path}（共 {len(losses)} 个点，"
          f"最近 loss={losses[-1]:.4f} @ step {steps[-1]}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
