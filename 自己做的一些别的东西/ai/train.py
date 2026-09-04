# train.py
import os
# 使用流式池化分配器：默认缓存分配器对“逐 token 递增大小”的分配无法复用旧块，
# 每条记录窗口增长期都会滞留一批递增块，跨记录线性累积导致显存爆炸（必须在 import torch 之前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
from contextlib import nullcontext
from tqdm import tqdm
import torch
import torch.nn.functional as F

from config import ModelArgs, TrainArgs
from model import WaveGaussianLM, ModelWrapper
from data import StreamingCLMDataset
from utils import save_checkpoint, load_checkpoint, count_parameters


def build_optimizers(model, args: TrainArgs):
    """参数分组：embed/head 与一维参数用 AdamW，其余二维权重用 Muon（use_muon 开关）"""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'embed' in name or 'head' in name or p.dim() < 2:
            adamw_params.append(p)
        else:
            muon_params.append(p)

    optimizers = {}
    if args.use_muon and muon_params:
        optimizers["muon"] = torch.optim.Muon(
            muon_params, lr=args.muon_lr, weight_decay=args.muon_weight_decay,
            momentum=args.muon_momentum, nesterov=args.muon_nesterov,
            ns_steps=args.muon_ns_steps,
        )
        print(f"Muon 优化 {sum(p.numel() for p in muon_params):,} 参数")
    else:
        adamw_params += muon_params
    optimizers["adamw"] = torch.optim.AdamW(
        adamw_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    print(f"AdamW 优化 {sum(p.numel() for p in adamw_params):,} 参数")
    return optimizers


def train(args: TrainArgs, model_args: ModelArgs):
    device = args.device
    latest_ckpt = os.path.join(args.output_dir, "latest.pt")

    # 数据集（固定顺序中文源，按条产出 token 流，训完即止）
    dataset = StreamingCLMDataset(args.data_dir, args.tokenizer_path)
    print(f"vocab_size: {model_args.vocab_size}")

    # 模型 / 优化器 / 外部状态（token 窗口 + 历史权重记忆）/ 混合精度
    model = WaveGaussianLM(
        vocab_size=model_args.vocab_size,
        d_model=model_args.d_model,
        num_layers=model_args.num_layers,
        first_gauss=model_args.first_gauss,
        first_wave=model_args.first_wave,
        layer_ff_dim=model_args.layer_ff_dim,
        layer_gauss=model_args.layer_gauss,
        layer_wave=model_args.layer_wave,
    ).to(device)
    print(f"总参数量: {count_parameters(model):,}")
    optimizers = build_optimizers(model, args)
    wrapper = ModelWrapper(model, max_input_len=model_args.max_input_len,
                           max_mem_len=model_args.max_mem_len)
    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if args.use_bf16 and "cuda" in device else nullcontext())

    # 断点续训：恢复模型 / 优化器 / 数据位置。
    start_step = 0
    if os.path.exists(latest_ckpt):
        start_step, _, data_state = load_checkpoint(
            latest_ckpt, model, optimizers)
        if data_state:
            dataset.resume(data_state)
            if dataset.is_exhausted():
                print("上次训练已跑完全部数据，如需继续请删除 checkpoint")
                return
            print("已恢复数据位置：断点所在记录将重训（偏差 ≤ 1 条）")
        print(f"已恢复训练：step {start_step}（记忆从空重新积累）")

    # loss 曲线数据：每 log_steps 步追加一个 (step, 平均 loss) 到 CSV，供 plot_loss.py 画图
    os.makedirs(args.output_dir, exist_ok=True)
    loss_log_path = os.path.join(args.output_dir, "train_loss.csv")
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w", encoding="utf-8") as f:
            f.write("step,loss\n")
    loss_sum, loss_count = 0.0, 0

    global_step, best_loss = start_step, float("inf")
    loss_val, done = float("inf"), False
    pbar = tqdm(total=dataset.total_tokens, unit="tok")

    for ids in dataset:
        # 每条记录从头开始：重置窗口与记忆（每次读一段话，初始一无所知）
        wrapper.reset()
        seq = torch.tensor(ids, dtype=torch.long, device=device)
        # 逐 token 流式：前向 → 反向 → 更新，计算图每步释放，显存与窗口无关
        for i in range(len(ids) - 1):
            with amp_ctx:
                logits = wrapper.step(ids[i])          # (1, vocab_size)
                loss = F.cross_entropy(logits, seq[i + 1:i + 2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for opt in optimizers.values():
                opt.step()
            model.zero_grad()

            global_step += 1
            loss_val = loss.item()

            # 累积 loss，攒满 log_steps 步写一个平均值到 CSV
            loss_sum += loss_val
            loss_count += 1
            if loss_count >= args.log_steps:
                with open(loss_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{global_step},{loss_sum / loss_count:.6f}\n")
                loss_sum, loss_count = 0.0, 0

            # 保存（含数据位置，随时可断点续训）
            if global_step % args.save_steps == 0:
                is_best = loss_val < best_loss
                best_loss = min(best_loss, loss_val)
                save_checkpoint(model, optimizers, global_step, loss_val, args,
                                is_best, data_state=dataset.get_state())

            if args.max_steps > 0 and global_step >= args.max_steps:
                done = True
                break
        pbar.update(len(ids))
        pbar.set_postfix(loss=f"{loss_val:.4f}")
        if done:
            break
    pbar.close()

    # 尾部不足一段的残余 loss 也写入（保证 CSV 始终含最新数据点）
    if loss_count > 0:
        with open(loss_log_path, "a", encoding="utf-8") as f:
            f.write(f"{global_step},{loss_sum / loss_count:.6f}\n")

    # 数据训完或 max_steps 提前停止：保存最终 checkpoint（含数据位置）
    save_checkpoint(model, optimizers, global_step, loss_val, args,
                    data_state=dataset.get_state())


if __name__ == "__main__":
    train(TrainArgs(), ModelArgs())
