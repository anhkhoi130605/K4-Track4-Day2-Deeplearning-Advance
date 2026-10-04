"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    # 0. Ánh xạ tên gọi tắt sang timm model name nếu có trong SUGGESTED_BACKBONES
    timm_name = SUGGESTED_BACKBONES.get(name, name)

    # 1. Xác định pretrained theo init
    is_pretrained = pretrained if init != "scratch" else False

    # 2. Tạo model với timm (tự động thay classification head mới với num_classes)
    model = timm.create_model(
        timm_name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )

    # 3. Ghi nhận tag trọng số thực tế (nếu có)
    if hasattr(model, "pretrained_cfg") and model.pretrained_cfg:
        tag = model.pretrained_cfg.get("tag", "default")
        print(f"[Model] {name} | Init: {init} | Pretrained: {is_pretrained} | Tag: {tag}")

    # 4. Nếu init == "frozen": đóng băng backbone, chỉ huấn luyện classifier head
    if init == "frozen":
        freeze_backbone(model)

    # 5. Thống kê số tham số trainable
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_p / total_p * 100) if total_p > 0 else 0
    print(f"  -> Trainable Params: {trainable_p / 1e6:.2f}M / {total_p / 1e6:.2f}M ({trainable_pct:.2f}%)")

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head.

    - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
    - lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
    """
    classifier = model.get_classifier()
    classifier_params = set(classifier.parameters())

    for param in model.parameters():
        if param in classifier_params:
            param.requires_grad = True
        else:
            param.requires_grad = False


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52:

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    classifier_params = set(classifier.parameters())

    backbone_weights = []
    backbone_bias_norm = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param in classifier_params:
            head_params.append(param)
        elif param.ndim <= 1 or "bn" in name or "norm" in name or "bias" in name:
            backbone_bias_norm.append(param)
        else:
            backbone_weights.append(param)

    groups = []
    if backbone_weights:
        groups.append({
            "params": backbone_weights,
            "lr": lr_backbone,
            "weight_decay": weight_decay
        })
    if backbone_bias_norm:
        groups.append({
            "params": backbone_bias_norm,
            "lr": lr_backbone,
            "weight_decay": 0.0  # Không áp dụng weight decay cho norm và bias
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay
        })

    return groups


def count_params(model: nn.Module, trainable_only: bool = False) -> float:
    """Số tham số (triệu). trainable_only=True chỉ đếm tham số trainable."""
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total = sum(p.numel() for p in model.parameters())
    return round(total / 1e6, 4)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x)."""
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return round(macs / 1e9, 4)
    except Exception as e:
        print(f"[Warning GMAC profile]: {e}")
        return 0.0


if __name__ == "__main__":
    print("=" * 70)
    print("KIEM TRA CAC BACKBONE GOI Y (PARAMS & GMACs):")
    print("=" * 70)

    for nick, model_name in SUGGESTED_BACKBONES.items():
        try:
            m = build_model(model_name, pretrained=False, num_classes=9, init="finetune")
            params = count_params(m)
            gmacs = count_gmacs(m, img_size=224)
            groups = param_groups(m, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
            print(f"[{nick:<16}] {model_name:<28} | Params: {params:>7.2f}M | GMACs: {gmacs:>6.2f} | Nhom params: {len(groups)}")
        except Exception as err:
            print(f"[{nick:<16}] Error: {err}")
