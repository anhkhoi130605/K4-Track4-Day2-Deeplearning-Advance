"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    """
    kind_lower = kind.lower()

    if kind_lower == "ce":
        return nn.CrossEntropyLoss(weight=kw.get("weight"))

    elif kind_lower in ["ls", "label_smoothing"]:
        smoothing = kw.get("smoothing", kw.get("label_smoothing", 0.1))
        return LabelSmoothingCE(smoothing=smoothing, weight=kw.get("weight"))

    elif kind_lower == "focal":
        gamma = kw.get("gamma", kw.get("focal_gamma", 2.0))
        alpha = kw.get("alpha", kw.get("weight"))
        return FocalLoss(gamma=gamma, alpha=alpha)

    elif kind_lower in ["ce_weighted", "weighted"]:
        weight = kw.get("weight")
        assert weight is not None, "ce_weighted yeu cau tham so weight!"
        return nn.CrossEntropyLoss(weight=weight)

    else:
        raise ValueError(f"Khong ho tro loai loss: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, f"smoothing phai thuoc [0, 1), nhan duoc {smoothing}"
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Neu smoothing == 0: tuong duong CE thuan
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, targets, weight=self.weight)

        num_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)

        # Tao one-hot smoothed target
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / num_classes)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + (self.smoothing / num_classes))

        if self.weight is not None:
            weight = self.weight.to(logits.device)
            true_dist = true_dist * weight.unsqueeze(0)

        loss = (-true_dist * log_preds).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    Bắt buộc kiểm tra: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        # Lay log_pt va pt tai dung nhan target
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)

        # Trong so focal: (1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        # Trong so theo lop alpha_t (neu co)
        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, targets)
            loss = at * loss

        return loss.mean()


def class_weights(counts: list | dict | np.ndarray, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp
    """
    if isinstance(counts, dict):
        # Sap xep theo class index 0, 1, ..., num_classes - 1
        num_classes = max(counts.keys()) + 1 if counts else 9
        counts_list = [counts.get(i, 0) for i in range(num_classes)]
    else:
        counts_list = list(counts)

    c_arr = np.array(counts_list, dtype=np.float64)
    c_arr = np.maximum(c_arr, 1.0)  # Tranh chia cho 0

    if beta <= 0.0:
        # Ti le nghich voi so anh (1 / n_c)
        weights = 1.0 / c_arr
    else:
        # Class-balanced loss (Cui et al. 2019)
        effective_num = 1.0 - np.power(beta, c_arr)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)

    # Chuan hoa tong trong so ve bang so lop (trung binh moi lop la 1.0)
    weights = weights / weights.sum() * len(c_arr)
    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]
    """
    if alpha > 0.0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]

    elif mode == "cutmix":
        _, _, h, w = x.shape

        # Tinh chieu rong va chieu cao hop cat theo dien tich (1 - lam)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Toa do tam ngau nhien
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        # Gioi han toa do hop cat nam trong bien anh
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

        # Dieu chinh lam theo dien tich thuc te
        actual_ratio = (x2 - x1) * (y2 - y1) / (w * h)
        lam = 1.0 - float(actual_ratio)

    else:
        raise ValueError(f"Khong ho tro mix mode: {mode}")

    return x_mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


if __name__ == "__main__":
    print("=" * 70)
    print("TEST CAC HAM TRONG losses.py:")
    print("=" * 70)

    # 1. Kiem tra FocalLoss(gamma=0) phai trung voi CrossEntropyLoss
    logits = torch.randn(8, 9)
    targets = torch.randint(0, 9, (8,))
    ce_val = F.cross_entropy(logits, targets).item()
    fl_val = FocalLoss(gamma=0.0)(logits, targets).item()
    print(f"1. Focal(gamma=0) vs CE: CE = {ce_val:.6f}, Focal = {fl_val:.6f} | Sai so = {abs(ce_val - fl_val):.2e}")
    assert abs(ce_val - fl_val) < 1e-6, "Loi: FocalLoss khi gamma=0 khong khop voi CE!"

    # 2. Kiem tra LabelSmoothingCE(smoothing=0)
    ls_val = LabelSmoothingCE(smoothing=0.0)(logits, targets).item()
    print(f"2. LabelSmoothing(eps=0) vs CE: CE = {ce_val:.6f}, LS = {ls_val:.6f} | Sai so = {abs(ce_val - ls_val):.2e}")
    assert abs(ce_val - ls_val) < 1e-6, "Loi: LabelSmoothing khi eps=0 khong khop voi CE!"

    # 3. Kiem tra class_weights
    dummy_counts = [1000, 500, 200, 100, 50, 20, 10, 5, 2]
    w_inv = class_weights(dummy_counts, beta=0.0)
    w_cb = class_weights(dummy_counts, beta=0.999)
    print(f"3. Class weights (beta=0, inv): {w_inv.numpy()[:3]}")
    print(f"   Class weights (beta=0.999, CB): {w_cb.numpy()[:3]}")

    # 4. Kiem tra Mixup va CutMix
    dummy_imgs = torch.randn(4, 3, 224, 224)
    dummy_lbls = torch.tensor([0, 1, 2, 3])
    x_mix, mix_targets = mix_batch(dummy_imgs, dummy_lbls, alpha=1.0, mode="mixup")
    print(f"4. Mixup output shape: {x_mix.shape}, lam = {mix_targets[2]:.4f}")

    x_cut, cut_targets = mix_batch(dummy_imgs, dummy_lbls, alpha=1.0, mode="cutmix")
    print(f"   CutMix output shape: {x_cut.shape}, lam = {cut_targets[2]:.4f}")

    crit = build_criterion("ce")
    loss = mixed_loss(crit, logits[:4], cut_targets)
    print(f"   Mixed loss value: {loss.item():.4f}")
    print("ALL TESTS PASSED IN losses.py!")
