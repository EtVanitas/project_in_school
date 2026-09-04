# utils.py
import os
import random
import torch
import numpy as np
from config import TrainArgs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer_muon, optimizer_adamw,
                    scheduler_muon, scheduler_adamw,
                    step, loss, args: TrainArgs, is_best=False):
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_muon_state_dict": optimizer_muon.state_dict(),
        "optimizer_adamw_state_dict": optimizer_adamw.state_dict(),
        "scheduler_muon_state_dict": scheduler_muon.state_dict(),
        "scheduler_adamw_state_dict": scheduler_adamw.state_dict(),
        "loss": loss,
    }
    path = os.path.join(args.output_dir, f"checkpoint-{step}.pt")
    torch.save(checkpoint, path)
    if is_best:
        best_path = os.path.join(args.output_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
    print(f"Checkpoint saved at step {step}")


def load_checkpoint(path, model, optimizer_muon, optimizer_adamw,
                     scheduler_muon, scheduler_adamw):
    if not os.path.exists(path):
        return 0, None
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_muon_state_dict" in checkpoint:
        optimizer_muon.load_state_dict(checkpoint["optimizer_muon_state_dict"])
        optimizer_adamw.load_state_dict(checkpoint["optimizer_adamw_state_dict"])
        scheduler_muon.load_state_dict(checkpoint["scheduler_muon_state_dict"])
        scheduler_adamw.load_state_dict(checkpoint["scheduler_adamw_state_dict"])
    return checkpoint["step"], checkpoint["loss"]


def count_parameters(model):
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
