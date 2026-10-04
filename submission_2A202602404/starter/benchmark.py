"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Giao diện bắt buộc:
    bench(fn, warmup, iters, sync) -> dict
    latency_report(model, batch_size, img_size, dtype, device, warmup, iters) -> dict
    tta_latency(model, k_views, ...) -> dict

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable
import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], any], warmup: int = 10, iters: int = 100, sync: Callable[[], None] | None = None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    Quy trình:
      1. Chạy `warmup` lần đầu rồi bỏ qua (làm nóng GPU, cache bộ nhớ, khởi tạo CUDA context).
      2. Với mỗi lần đo trong `iters`:
         - sync()
         - t0 = time.perf_counter()
         - fn()
         - sync()
         - tính thời gian mili-giây: (time.perf_counter() - t0) * 1000.0
      3. Tính các phân vị p50, p95, p99 và mean bằng numpy.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Lặp đo thời gian
    latencies = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()

        fn()

        if sync is not None:
            sync()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)

    times = np.array(latencies, dtype=np.float64)

    return {
        "p50": round(float(np.percentile(times, 50)), 3),
        "p95": round(float(np.percentile(times, 95)), 3),
        "p99": round(float(np.percentile(times, 99)), 3),
        "mean": round(float(np.mean(times)), 3),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    Lưu ý kỹ thuật (slide trang 73):
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - lấy tên GPU bằng torch.cuda.get_device_name()
      - ở batch 1, AMP có thể CHẬM hơn FP32 do overhead kernel casting.
    """
    # 1. Xác định thiết bị
    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    target_device = torch.device(device if use_cuda else "cpu")
    gpu_name = torch.cuda.get_device_name(target_device) if use_cuda else "CPU"
    sync_fn = torch.cuda.synchronize if use_cuda else None

    # 2. Chuẩn bị model ở chế độ eval
    model.eval()
    model.to(target_device)

    # 3. Chuẩn bị đầu vào và hàm forward theo dtype
    dtype_lower = dtype.lower()

    if dtype_lower == "fp16":
        # Chuyển toàn bộ model và tensor sang half (fp16)
        model_eval = model.half()
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device, dtype=torch.float16)

        def forward_fn():
            with torch.inference_mode():
                return model_eval(dummy_input)

    elif dtype_lower == "amp":
        model_eval = model.float()
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device, dtype=torch.float32)
        amp_dtype = torch.float16 if use_cuda else torch.bfloat16

        def forward_fn():
            with torch.inference_mode(), torch.autocast(device_type=target_device.type, dtype=amp_dtype):
                return model_eval(dummy_input)

    else:  # Mặc định là fp32
        model_eval = model.float()
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device, dtype=torch.float32)

        def forward_fn():
            with torch.inference_mode():
                return model_eval(dummy_input)

    # 4. Thực hiện benchmark
    stats = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    # 5. Tính thông lượng Throughput (ảnh / giây) dựa trên p50
    p50_ms = stats["p50"]
    images_per_s = round(batch_size / (p50_ms / 1000.0), 2) if p50_ms > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype.upper(),
        "batch": batch_size,
        "img_size": img_size,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "mean": stats["mean"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ của TTA (Test-Time Augmentation) K view (slide trang 63 & 75).

    TTA K view chạy K biến thể của mỗi ảnh (ví dụ ảnh gốc + ảnh lật ngang, crop)
    rồi gộp logit. Chi phí lý thuyết xấp xỉ K * p50 của 1 view.
    """
    # 1. Đo độ trễ 1-view làm mốc chuẩn
    single_res = latency_report(
        model,
        batch_size=batch_size,
        img_size=img_size,
        dtype=dtype,
        device=device,
        warmup=warmup,
        iters=iters
    )

    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    target_device = torch.device(device if use_cuda else "cpu")
    sync_fn = torch.cuda.synchronize if use_cuda else None

    # 2. Tạo hàm forward TTA với K views
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device)
    model.eval()
    model.to(target_device)

    def tta_forward():
        with torch.inference_mode():
            # Mô phỏng K views: view 0 là ảnh gốc, các view tiếp theo là flip hoặc biến thể
            views = [dummy_input] + [torch.flip(dummy_input, dims=[-1]) for _ in range(k_views - 1)]
            logits = [model(v) for v in views]
            # Gộp logits trung bình
            return torch.stack(logits).mean(dim=0)

    tta_stats = bench(tta_forward, warmup=warmup, iters=iters, sync=sync_fn)

    expected_p50 = round(single_res["p50"] * k_views, 3)
    ratio = round(tta_stats["p50"] / single_res["p50"], 2) if single_res["p50"] > 0 else 0.0

    return {
        "k_views": k_views,
        "single_p50_ms": single_res["p50"],
        "tta_p50_ms": tta_stats["p50"],
        "expected_k_p50_ms": expected_p50,
        "ratio_actual_vs_single": ratio,
        "gpu": single_res["gpu"],
        "dtype": dtype.upper(),
        "batch": batch_size,
        "img_size": img_size,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("DEMO KIEM TRA LATENCY BENCHMARK:")
    print("=" * 70)

    try:
        import timm
        # Thử nghiệm với resnet18 để đo nhanh
        test_model = timm.create_model("resnet18", pretrained=False, num_classes=9)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Thiet bi do: {device.upper()}")
        rep = latency_report(test_model, batch_size=1, img_size=224, dtype="fp32", device=device, warmup=5, iters=20)
        print("Latency Report (Batch=1, FP32):")
        for k, v in rep.items():
            print(f"  {k}: {v}")

        tta_rep = tta_latency(test_model, k_views=2, batch_size=1, img_size=224, device=device, warmup=5, iters=10)
        print("\nTTA Latency (K=2 views):")
        for k, v in tta_rep.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Loi demo: {e}")
