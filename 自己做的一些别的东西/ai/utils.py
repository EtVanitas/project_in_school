# utils.py
import os
import glob
import torch
from config import TrainArgs


def save_checkpoint(model, optimizers: dict,
                    step, loss, args: TrainArgs, is_best=False,
                    data_state=None):
    """保存 checkpoint，同时写入 latest.pt 便于断点续训。"""
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
        "loss": loss,
    }
    if data_state is not None:
        checkpoint["data_state"] = data_state
    path = os.path.join(args.output_dir, f"checkpoint-{step}.pt")
    torch.save(checkpoint, path)
    torch.save(checkpoint, os.path.join(args.output_dir, "latest.pt"))
    if is_best:
        best_path = os.path.join(args.output_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
    for old in glob.glob(os.path.join(args.output_dir, "checkpoint-*.pt")):
        if old != path:
            os.remove(old)
    print(f"Checkpoint saved at step {step}")


def load_checkpoint(path, model, optimizers: dict):
    """加载 checkpoint，返回 (step, loss, data_state)。"""
    if not os.path.exists(path):
        return 0, None, None
    checkpoint = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        print(f"警告：模型参数与 checkpoint 不匹配（{e}），跳过模型权重加载")

    if "optimizers" in checkpoint:
        for k, opt in optimizers.items():
            if k in checkpoint["optimizers"]:
                try:
                    opt.load_state_dict(checkpoint["optimizers"][k])
                except (RuntimeError, ValueError) as e:
                    print(f"警告：优化器 {k} 状态不兼容（{e}），跳过加载")
    return (checkpoint.get("step", 0), checkpoint.get("loss"),
            checkpoint.get("data_state"))


def count_parameters(model):
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
