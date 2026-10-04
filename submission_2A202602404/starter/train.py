"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Chạy thí nghiệm:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
    hoặc dùng cờ trực tiếp:
    python train.py --exp_id B01 --backbone resnet50 --seed 0 --epochs 12
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, fields
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# Toi uu hoa phan bo VRAM tren GPU 6GB laptop va ma hoa UTF-8 cho Windows
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure') and sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# Đảm bảo import được eval.py ở thư mục gốc và các module trong starter
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

STARTER_DIR = Path(__file__).resolve().parent
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from eval import compute_metrics, save_predictions
from dataset import load_split, check_split, build_transforms, make_loader
from model import build_model, freeze_backbone, param_groups, count_params, count_gmacs
from losses import build_criterion, class_weights, mix_batch, mixed_loss


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 32              # 32 hoặc 64 tuỳ bộ nhớ VRAM GPU
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = str(STARTER_DIR / "data" / "images")
    labels_dir: str = str(STARTER_DIR / "data" / "labels")
    out_dir: str = str(STARTER_DIR / "tensorboard")
    checkpoint_dir: str = str(STARTER_DIR / "trained_models")
    pred_dir: str = str(STARTER_DIR / "predictions")
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # --- checkpoint & resume ---
    save_checkpoint: bool = True
    resume: str | None = None


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def get_checkpoint_dir(cfg: Config) -> Path:
    """Thư mục lưu weights/checkpoint: <checkpoint_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.checkpoint_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên để tái lập thí nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0 (slide Day 2, trang 55)."""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay từ 1.0 về 0.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Nạp trọng số EMA vào mô hình để đánh giá."""
        self.backup = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Khôi phục lại trọng số gốc của mô hình."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None,
                    writer: SummaryWriter | None = None, epoch: int = 0) -> dict:
    """Một epoch huấn luyện với progress bar tqdm và TensorBoard log."""
    model.train()

    # Lưu ý (GUIDE.md mục 3.2): nếu init == "frozen": giữ backbone và BatchNorm ở chế độ eval
    if cfg.init == "frozen":
        freeze_backbone(model)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    total_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1:02d}/{cfg.epochs:02d} [Train]", leave=False, colour="cyan", disable=cfg.report_progress)
    use_amp = cfg.amp and device.type == "cuda"

    for step, batch in enumerate(pbar if not cfg.report_progress else loader):
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        # Trộn mẫu nếu cấu hình mix (Mixup / CutMix)
        if cfg.mix and cfg.mix in ["mixup", "cutmix"]:
            images, mix_targets = mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)
        else:
            mix_targets = None

        optimizer.zero_grad()

        # Forward với AMP
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if mix_targets is not None:
                loss = mixed_loss(criterion, logits, mix_targets)
            else:
                loss = criterion(logits, labels)

        # Backward và cập nhật trọng số
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        curr_lr = optimizer.param_groups[0]["lr"]

        if not cfg.report_progress:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{curr_lr:.2e}"})
        else:
            print(f"[PBAR] phase=train epoch={epoch + 1} epochs={cfg.epochs} step={step + 1} total={num_batches} loss={loss.item():.4f} lr={curr_lr:.2e}", flush=True)

        # Ghi log step vào TensorBoard
        if writer is not None and step % 20 == 0:
            global_step = epoch * num_batches + step
            writer.add_scalar("Train/Step_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Step_LR", curr_lr, global_step)

    avg_loss = total_loss / max(1, num_batches)
    return {"train_loss": avg_loss, "lr": curr_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device, report_progress: bool = False):
    """Chạy model trên loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc="[Evaluate]", leave=False, colour="cyan", disable=report_progress)
    with torch.inference_mode():
        for step, batch in enumerate(pbar if not report_progress else loader):
            images, labels, filenames = batch
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            if report_progress and ((step + 1) % 5 == 0 or (step + 1) == num_batches):
                print(f"[PBAR] phase=val step={step + 1} total={num_batches}", flush=True)

            all_filenames.extend(filenames)
            if isinstance(labels, torch.Tensor):
                all_targets.extend(labels.cpu().numpy().tolist())
            else:
                all_targets.extend(labels)

            all_logits.append(logits.cpu().float().numpy())

    y_true = np.array(all_targets, dtype=np.int64)
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 9), dtype=np.float32)
    avg_loss = total_loss / max(1, num_batches)
    return all_filenames, y_true, logits_arr, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training -> curves/<exp_id>_<backbone>.png (GUIDE.md mục 6.2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Loss qua các Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Macro-F1
    if "val_macro_f1" in df.columns:
        axes[1].plot(df["epoch"], df["val_macro_f1"], label="Val Macro-F1", color="green", marker="^")
        axes[1].set_title("Macro-F1 (Tập Val)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Learning Rate
    if "lr" in df.columns:
        axes[2].plot(df["epoch"], df["lr"], label="Learning Rate", color="orange", marker=".")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi kết quả cần thiết."""
    print("\n" + "=" * 75)
    print(f"BAT DAU THI NGHIEM: {cfg.exp_id} | Backbone: {cfg.backbone} | Seed: {cfg.seed}")
    print("=" * 75)

    # 1. Cố định seed & tạo thư mục lưu kết quả và weights
    set_seed(cfg.seed)
    output_dir = run_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = get_checkpoint_dir(cfg)
    if cfg.save_checkpoint:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)

    # Ghi config.json
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Khởi tạo TensorBoard SummaryWriter
    tb_log_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard log luu tai: {tb_log_dir}")
    if cfg.save_checkpoint:
        print(f"Thu muc luu weights model (trained_models): {ckpt_dir}")

    # 2. Đọc và kiểm tra split dữ liệu
    train_df, val_df, test_df = load_split(cfg.labels_dir, fold=cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. Dựng DataLoader
    train_transform = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_transform = build_transforms(train=False, img_size=cfg.img_size)

    train_loader = make_loader(
        train_df, cfg.images_dir, train_transform, cfg.batch_size,
        train=True, sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = make_loader(
        val_df, cfg.images_dir, val_transform, cfg.batch_size,
        train=False, num_workers=cfg.num_workers
    )

    # 4. Thiết bị, Mô hình, Loss, Optimizer, Scheduler, Scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Su dung thiet bi: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Xử lý trọng số loss nếu dùng ce_weighted
    weight_tensor = None
    if cfg.loss in ["ce_weighted", "weighted"]:
        counts_dict = train_df["Label"].value_counts().to_dict()
        weight_tensor = class_weights(counts_dict, beta=cfg.class_weight_beta or 0.0)

    criterion = build_criterion(
        kind=cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=weight_tensor
    )

    model = build_model(
        name=cfg.backbone,
        pretrained=(cfg.init != "scratch"),
        num_classes=9,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    )
    model.to(device)

    total_params = count_params(model)
    gmacs = count_gmacs(model, cfg.img_size)
    print(f"Tham so mo hinh: {total_params}M | GMACs: {gmacs}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    # 5. Vòng lặp huấn luyện từng Epoch
    history: list[dict] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_val_logits = None
    best_val_probs = None
    best_val_fns = None
    best_val_y_true = None
    start_epoch = 0

    # Nạp checkpoint nếu có cờ --resume
    if cfg.resume:
        resume_file = Path(cfg.resume)
        if not resume_file.is_absolute() and not resume_file.exists():
            candidate = Path(cfg.checkpoint_dir) / cfg.exp_id / f"seed{cfg.seed}" / resume_file
            if candidate.exists():
                resume_file = candidate
        if resume_file.exists():
            print(f"[*] Nap checkpoint resume tu: {resume_file}")
            ckpt = torch.load(resume_file, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"[!] Canh bao khong nap duoc optimizer: {e}")
            start_epoch = ckpt.get("epoch", 0)
            best_macro_f1 = ckpt.get("macro_f1", -1.0)
            best_epoch = start_epoch
            print(f"[*] Tiep tuc train tu Epoch {start_epoch + 1}/{cfg.epochs} (Best F1: {best_macro_f1:.4f})")
            if (output_dir / "history.csv").exists():
                try:
                    history = pd.read_csv(output_dir / "history.csv").to_dict(orient="records")
                except Exception:
                    pass
        else:
            print(f"[!] Khong tim thay checkpoint: {cfg.resume}")

    for epoch in range(start_epoch, cfg.epochs):
        t_start = time.time()

        # Huấn luyện 1 epoch
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            cfg, device, ema=ema, writer=writer, epoch=epoch
        )

        # Đánh giá trên tập VAL (nếu có EMA: áp dụng EMA để đánh giá)
        if ema is not None:
            ema.apply_shadow(model)

        val_fns, val_y_true, val_logits, val_loss = evaluate(model, val_loader, criterion, device, report_progress=cfg.report_progress)

        if ema is not None:
            ema.restore(model)

        # Tính toán chỉ số chính thức bằng eval.compute_metrics
        probs = _softmax_np(val_logits)
        metrics = compute_metrics(val_y_true, probs.argmax(axis=1), probs)
        val_macro_f1 = metrics["macro_f1"]
        val_top1 = metrics["top1"]
        epoch_time = time.time() - t_start

        # Ghi log epoch vào TensorBoard
        writer.add_scalar("Epoch/Train_Loss", train_res["train_loss"], epoch + 1)
        writer.add_scalar("Epoch/Val_Loss", val_loss, epoch + 1)
        writer.add_scalar("Epoch/Val_MacroF1", val_macro_f1, epoch + 1)
        writer.add_scalar("Epoch/Val_Top1", val_top1, epoch + 1)
        writer.add_scalar("Epoch/LR", train_res["lr"], epoch + 1)

        if cfg.report_progress:
            print(f"[PBAR] phase=epoch_end epoch={epoch + 1} epochs={cfg.epochs} val_f1={val_macro_f1:.4f} val_loss={val_loss:.4f} lr={train_res['lr']:.2e}", flush=True)

        print(f"Epoch {epoch + 1:02d}/{cfg.epochs:02d} | Train Loss: {train_res['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Top-1: {val_top1:.4f} | "
              f"Time: {epoch_time:.1f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_res["train_loss"], 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_top1": round(val_top1, 4),
            "lr": train_res["lr"],
            "time_s": round(epoch_time, 2)
        })

        # 1. Luôn lưu checkpoint của epoch mới nhất vào last.pt nếu bật save_checkpoint
        if cfg.save_checkpoint:
            last_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": val_macro_f1,
                "cfg": asdict(cfg),
            }
            torch.save(last_checkpoint, ckpt_dir / "last.pt")
            torch.save(last_checkpoint, output_dir / "last.pt")

        # 2. Cập nhật kết quả tốt nhất theo Macro-F1 val (hòa thì lấy epoch sớm hơn)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            best_val_logits = val_logits
            best_val_probs = probs
            best_val_fns = val_fns
            best_val_y_true = val_y_true

            if cfg.save_checkpoint:
                best_checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_f1": best_macro_f1,
                    "cfg": asdict(cfg),
                }
                torch.save(best_checkpoint, ckpt_dir / "best.pt")
                torch.save(best_checkpoint, output_dir / "best.pt")
                torch.save(best_checkpoint, ckpt_dir / "best_model.pth")
                torch.save(best_checkpoint, output_dir / "best_model.pth")

    if cfg.save_checkpoint:
        print(f"\n=> Checkpoint tot nhat tai Epoch {best_epoch} (Val Macro-F1 = {best_macro_f1:.4f}) -> Da luu: {ckpt_dir / 'best.pt'}")
        print(f"=> Checkpoint cuoi cung -> Da luu: {ckpt_dir / 'last.pt'}")
    else:
        print(f"\n=> Ket qua tot nhat tai Epoch {best_epoch} (Val Macro-F1 = {best_macro_f1:.4f}) [Khong luu checkpoint file]")

    # 6. Ghi dự đoán trên tập VAL đúng chuẩn eval.py
    if best_val_probs is not None:
        save_predictions(pred_path(cfg, "val"), best_val_fns, best_val_y_true, best_val_probs)
        np.save(output_dir / "val_logits.npy", best_val_logits)
        print(f"Da luu du doan VAL tai: {pred_path(cfg, 'val')}")

    # 7. NẾU bật save_test_predictions (chỉ ở Bước 4 - Chung kết):
    if cfg.save_test_predictions:
        print("\n[Buoc 4] Dang danh gia tren tap TEST...")
        test_loader = make_loader(
            test_df, cfg.images_dir, val_transform, cfg.batch_size,
            train=False, num_workers=cfg.num_workers
        )
        # Nạp lại checkpoint tốt nhất (best.pt)
        ckpt_file = ckpt_dir / "best.pt"
        if not ckpt_file.exists():
            ckpt_file = output_dir / "best.pt"
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_fns, test_y_true, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = _softmax_np(test_logits)

        save_predictions(pred_path(cfg, "test"), test_fns, test_y_true, test_probs)
        np.save(output_dir / "test_logits.npy", test_logits)
        print(f"Da luu du doan TEST tai: {pred_path(cfg, 'test')}")

    # 8. Ghi history.csv và vẽ đồ thị curves
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    curves_dir = Path("curves")
    curves_dir.mkdir(parents=True, exist_ok=True)
    curve_file = curves_dir / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_file, f"Thi nghiem {cfg.exp_id} ({cfg.backbone})")
    print(f"Da luu bieu do training tai: {curve_file}")

    writer.close()

    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "val_macro_f1": round(best_macro_f1, 4),
        "params_m": total_params,
        "gmacs": gmacs,
        "run_dir": str(output_dir),
    }
    print(f"Ket qua tom tat: {summary}\n")
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal'] thành dict, ép kiểu theo Config."""
    overrides = {}
    config_fields = {f.name: f.type for f in fields(Config)}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không đúng định dạng KEY=VALUE: {pair}")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in config_fields:
            raise ValueError(f"Khóa '{key}' không tồn tại trong Config! Các khóa hợp lệ: {list(config_fields.keys())}")

        target_type = config_fields[key]
        if val.lower() == "none":
            overrides[key] = None
        elif target_type == bool or val.lower() in ["true", "false"]:
            overrides[key] = (val.lower() == "true")
        elif target_type == int:
            overrides[key] = int(val)
        elif target_type == float:
            overrides[key] = float(val)
        else:
            overrides[key] = val

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepWeeds Lab Day 2 Training Pipeline")

    # Cho phep ghi de linh hoat bang --set KEY=VALUE ...
    parser.add_argument("--set", nargs="*", default=[], help="Ghi de cau hinh: --set exp_id=B01 backbone=resnet50 seed=0")

    # Cac co dong lenh thong dung
    parser.add_argument("--exp_id", type=str, default=None, help="Ma thi nghiem: B01, T00, F01...")
    parser.add_argument("--backbone", type=str, default=None, help="Ten backbone timm")
    parser.add_argument("--init", type=str, default=None, choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--epochs", type=int, default=None, help="So epoch huan luyen")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr_backbone", type=float, default=None, help="Learning rate cho backbone")
    parser.add_argument("--lr_head", type=float, default=None, help="Learning rate cho head")
    parser.add_argument("--aug", type=str, default=None, help="Muc augmentation: basic, color, randaug, trivial")
    parser.add_argument("--loss", type=str, default=None, help="Ham loss: ce, ls, focal, ce_weighted")
    parser.add_argument("--sampler", type=str, default=None, help="Can bang mau: None hoac balanced")
    parser.add_argument("--mix", type=str, default=None, help="Tron mau: None, mixup, cutmix")
    parser.add_argument("--seed", type=int, default=None, help="Seed ngau nhien")
    parser.add_argument("--save_test_predictions", action="store_true", help="Ghi du doan tren TEST (chi dung o Buoc 4)")
    parser.add_argument("--save_checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Luu checkpoint model (best.pt, last.pt). Dung --no-save-checkpoint de tat")
    parser.add_argument("--resume", type=str, default=None, help="Duong dan checkpoint .pt de train tiep")
    parser.add_argument("--num_workers", type=int, default=None, help="So luong worker cho DataLoader (mac dinh: 2)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Thu muc luu weights checkpoint (mac dinh: starter/trained_models)")
    parser.add_argument("--out_dir", type=str, default=None, help="Thu muc luu ket qua tong the (mac dinh: starter/runs)")
    parser.add_argument("--report_progress", action="store_true", help="Phat tien do format [PBAR] cho multi-runner")

    args = parser.parse_args()

    cfg = Config()

    # 1. Áp dụng các cờ trực tiếp
    for field_name in ["exp_id", "backbone", "init", "epochs", "batch_size", "lr_backbone",
                       "lr_head", "aug", "loss", "sampler", "mix", "seed", "save_checkpoint",
                       "resume", "num_workers", "checkpoint_dir", "out_dir", "report_progress"]:
        val = getattr(args, field_name)
        if val is not None:
            setattr(cfg, field_name, val)

    if args.save_test_predictions:
        cfg.save_test_predictions = True

    # 2. Áp dụng --set nếu có
    if args.set:
        overrides = parse_overrides(args.set)
        for k, v in overrides.items():
            setattr(cfg, k, v)

    run(cfg)


if __name__ == "__main__":
    main()
