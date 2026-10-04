"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bắt buộc:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model: nn.Module, loader, device: str = "cuda", view: Callable | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Trả về: (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9]).
    """
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    target_device = torch.device(device if use_cuda else "cpu")

    model.eval()
    model.to(target_device)

    all_filenames: list[str] = []
    all_targets: list[int] = []
    all_logits: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            images, labels, filenames = batch
            images = images.to(target_device)

            if view is not None:
                images = view(images)

            logits = model(images)

            all_filenames.extend(filenames)
            if isinstance(labels, torch.Tensor):
                all_targets.extend(labels.cpu().numpy().tolist())
            else:
                all_targets.extend(labels)

            all_logits.append(logits.cpu().float().numpy())

    y_true = np.array(all_targets, dtype=np.int64)
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 9), dtype=np.float32)

    return all_filenames, y_true, logits_arr


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Giữ nguyên ảnh gốc."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) trên chiều rộng cuối cùng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> list[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop x crop`.

    Trả về list gồm 5 batch tensors.
    """
    _, _, h, w = x.shape
    assert h >= crop and w >= crop, f"Kích thước ảnh ({h}, {w}) nhỏ hơn kích thước crop {crop}!"

    # 4 góc và 1 giữa
    tl = x[:, :, 0:crop, 0:crop]
    tr = x[:, :, 0:crop, w - crop:w]
    bl = x[:, :, h - crop:h, 0:crop]
    br = x[:, :, h - crop:h, w - crop:w]

    center_y = (h - crop) // 2
    center_x = (w - crop) // 2
    center = x[:, :, center_y:center_y + crop, center_x:center_x + crop]

    return [center, tl, tr, bl, br]


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`. Trả về list các batch."""
    scaled_batches = []
    for s in sizes:
        scaled = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        scaled_batches.append(scaled)
    return scaled_batches


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Softmax an toàn số học trên numpy."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một ma trận xác suất (N, 9) (slide trang 62).

    - space="prob":  trung bình xác suất softmax của từng view
    - space="logit": trung bình logit rồi mới tính softmax
    """
    assert len(logits_per_view) > 0, "Danh sách logits_per_view không được rỗng!"

    if space == "prob":
        probs_list = [_softmax_np(lg) for lg in logits_per_view]
        return np.mean(probs_list, axis=0)

    elif space == "logit":
        mean_logits = np.mean(logits_per_view, axis=0)
        return _softmax_np(mean_logits)

    else:
        raise ValueError(f"Không hỗ trợ space={space}. Chọn 'prob' hoặc 'logit'.")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed) (slide trang 67).

    Chi phí suy luận = tổng số mô hình. Yêu cầu cùng tập ảnh và cùng thứ tự file.
    """
    assert len(list_of_probs) > 0, "Danh sách xác suất rỗng!"
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray | torch.Tensor, val_labels: np.ndarray | torch.Tensor) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu hàm mất mát NLL trên tập VAL (slide trang 69).

    Lưu ý: KHÔNG khớp T trên test. Độ chính xác (Accuracy/F1) không đổi vì thứ tự logit giữ nguyên.
    """
    if isinstance(val_logits, np.ndarray):
        logits = torch.from_numpy(val_logits).float()
    else:
        logits = val_logits.float()

    if isinstance(val_labels, np.ndarray):
        labels = torch.from_numpy(val_labels).long()
    else:
        labels = val_labels.long()

    # Khởi tạo log(T) = 0 (tương đương T = 1.0)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def eval_nll():
        optimizer.zero_grad()
        T = torch.exp(log_t)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(eval_nll)
    best_t = float(torch.exp(log_t).item())
    return round(best_t, 4)


def apply_temperature(logits: np.ndarray | torch.Tensor, T: float) -> np.ndarray:
    """Trả về xác suất softmax(logits / T)."""
    t_val = max(float(T), 1e-4)

    if isinstance(logits, torch.Tensor):
        scaled = logits / t_val
        return F.softmax(scaled, dim=-1).cpu().numpy()
    else:
        scaled = logits / t_val
        return _softmax_np(scaled)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước (slide trang 71 & 75).

    Công thức:
        w' = gamma * w / sqrt(var + eps)
        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Kiểm tra: Sai số đầu ra trước và sau gộp < 1e-4.
    """
    model_fused = copy.deepcopy(model).eval()

    def _fuse_recursive(module: nn.Module):
        prev_name = None
        prev_child = None

        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d) and isinstance(prev_child, nn.Conv2d):
                conv = prev_child
                bn = child

                gamma = bn.weight
                beta = bn.bias
                mean = bn.running_mean
                var = bn.running_var
                eps = bn.eps

                std = torch.sqrt(var + eps)
                w = conv.weight
                b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)

                # Áp dụng công thức gộp trọng số
                w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
                b_fused = beta + gamma * (b - mean) / std

                # Tạo Conv2d mới có bias
                new_conv = nn.Conv2d(
                    conv.in_channels,
                    conv.out_channels,
                    conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                    bias=True,
                    padding_mode=conv.padding_mode
                )
                new_conv.weight.data.copy_(w_fused)
                new_conv.bias.data.copy_(b_fused)

                setattr(module, prev_name, new_conv)
                setattr(module, name, nn.Identity())

                prev_child = None
                prev_name = None
            else:
                _fuse_recursive(child)
                prev_name = name
                prev_child = child

    _fuse_recursive(model_fused)
    return model_fused


if __name__ == "__main__":
    print("=" * 70)
    print("KIEM TRA CAC PHUONG PHAP SUY LUAN TRONG inference.py:")
    print("=" * 70)

    # 1. Kiem tra TTA views
    dummy_x = torch.randn(2, 3, 224, 224)
    x_flip = view_hflip(dummy_x)
    print(f"1. View HFlip: Shape {x_flip.shape}")

    crops = views_multicrop(dummy_x, crop=200)
    print(f"2. View Multicrop (5 crops): {len(crops)} crops, moi crop shape {crops[0].shape}")

    scales = views_multiscale(dummy_x, sizes=[224, 256])
    print(f"3. View Multiscale: {len(scales)} scales ({[tuple(s.shape) for s in scales]})")

    # 2. Kiem tra Temperature Scaling
    val_logits = np.random.randn(50, 9) * 2.0
    val_labels = np.random.randint(0, 9, size=50)
    opt_t = fit_temperature(val_logits, val_labels)
    calibrated_probs = apply_temperature(val_logits, opt_t)
    print(f"4. Temperature Scaling: T toi uu = {opt_t:.4f}, Sum probs = {calibrated_probs[0].sum():.4f}")

    # 3. Kiem tra fuse_conv_bn
    from torchvision.models import resnet18
    r18 = resnet18().eval()
    with torch.inference_mode():
        orig_out = r18(dummy_x)
        fused_r18 = fuse_conv_bn(r18)
        fused_out = fused_r18(dummy_x)
        max_diff = (orig_out - fused_out).abs().max().item()
    print(f"5. Fuse Conv-BN: Sai so lon nhat = {max_diff:.2e} (Yeu cau < 1e-4)")
    assert max_diff < 1e-4, "Loi: Sai so gop Conv-BN qua lon!"

    print("ALL TESTS PASSED IN inference.py!")
