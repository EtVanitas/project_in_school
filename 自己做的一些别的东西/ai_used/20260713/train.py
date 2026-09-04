# train.py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from config import ModelArgs, TrainArgs
from model import TransformerForMLM
from data import StreamingMLMDataset
from utils import set_seed, save_checkpoint, load_checkpoint, count_parameters


def train(args: TrainArgs, model_args: ModelArgs):
    device = args.device
    set_seed(args.seed)

    # 断点续训：检查是否存在已有 checkpoint
    latest_ckpt = os.path.join(args.output_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        print(f"发现断点，将恢复训练: {latest_ckpt}")

    # 流式数据集
    dataset = StreamingMLMDataset(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        max_len=model_args.max_seq_len,
        mlm_prob=0.15,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,              # IterableDataset 不支持多进程
        pin_memory=True,
    )

    # 模型
    model = TransformerForMLM(model_args)
    model.to(device)
    print(f"总参数量: {count_parameters(model):,}")

    # 参数分组
    muon_params = []      # 二维权重矩阵（除嵌入层和 lm_head 外）
    adamw_params = []     # 一维参数（偏置、Norm 权重）+ 嵌入层 + lm_head

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'tok_embeddings' in name or 'lm_head' in name:
            adamw_params.append(p)
        elif p.dim() < 2:
            adamw_params.append(p)
        else:
            muon_params.append(p)

    # 创建两个优化器
    optimizer_muon = torch.optim.Muon(
        muon_params,
        lr=args.muon_lr,
        weight_decay=args.muon_weight_decay,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )

    optimizer_adamw = torch.optim.AdamW(
        adamw_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # 分别为两个优化器创建调度器（步数相同）
    total_steps = args.max_steps if args.max_steps > 0 else 1_000_000

    scheduler_muon = get_linear_schedule_with_warmup(
        optimizer_muon,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    scheduler_adamw = get_linear_schedule_with_warmup(
        optimizer_adamw,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # 加载断点（必须在 scheduler 创建之后）
    start_step = 0
    if os.path.exists(latest_ckpt):
        start_step, _ = load_checkpoint(
            latest_ckpt, model, optimizer_muon, optimizer_adamw,
            scheduler_muon, scheduler_adamw)
        print(f"已恢复训练：step {start_step}")

    # 日志
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    global_step = start_step
    best_loss = float("inf")
    model.zero_grad()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if args.use_bf16 and device == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, loss = model(input_ids, labels=labels)
            else:
                logits, loss = model(input_ids, labels=labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 分别对两个优化器执行 step 和调度器执行 step
            optimizer_muon.step()
            optimizer_adamw.step()
            scheduler_muon.step()
            scheduler_adamw.step()
            optimizer_muon.zero_grad()
            optimizer_adamw.zero_grad()

            global_step += 1

            # 日志
            if global_step % args.log_steps == 0:
                current_loss = loss.item()
                lr_muon = scheduler_muon.get_last_lr()[0]
                lr_adamw = scheduler_adamw.get_last_lr()[0]
                print(f"Step {global_step}: loss = {current_loss:.4f}, lr_muon = {lr_muon:.2e}, lr_adamw = {lr_adamw:.2e}")
                writer.add_scalar("loss", current_loss, global_step)
                writer.add_scalar("lr_muon", lr_muon, global_step)
                writer.add_scalar("lr_adamw", lr_adamw, global_step)
                

            if global_step % args.save_steps == 0:
                save_checkpoint(
                    model, optimizer_muon, optimizer_adamw,
                    scheduler_muon, scheduler_adamw,
                    global_step, loss.item(), args)
                # 同时保存为 latest.pt 便于断点续训
                ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                latest_path = os.path.join(args.output_dir, "latest.pt")
                torch.save(torch.load(ckpt_path, map_location="cpu"), latest_path)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_checkpoint(
                        model, optimizer_muon, optimizer_adamw,
                        scheduler_muon, scheduler_adamw,
                        global_step, loss.item(), args, is_best=True)

            # max_steps > 0 时受步数限制
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # 最终保存
    save_checkpoint(
        model, optimizer_muon, optimizer_adamw,
        scheduler_muon, scheduler_adamw,
        global_step, loss.item(), args, is_best=False)
    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
    latest_path = os.path.join(args.output_dir, "latest.pt")
    torch.save(torch.load(ckpt_path, map_location="cpu"), latest_path)

if __name__ == "__main__":
    # 示例：合并配置并启动训练
    model_args = ModelArgs()
    train_args = TrainArgs()
    train(train_args, model_args)
