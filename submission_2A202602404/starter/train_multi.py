"""train_multi.py - Multi-model runner với tqdm màu cyan cho từng model song song.

Tối ưu cho cấu hình:
  - GPU: NVIDIA RTX 3050 6GB Laptop GPU (VRAM 6.0 GB)
  - RAM: 16 GB (còn trống ~3.8 GB)

Đặc điểm nổi bật:
  - 3 thanh tqdm màu cyan xếp tầng trực tiếp trên terminal (position 0, 1, 2).
  - Cập nhật tiến độ loss, lr, epoch theo thời gian thực cho từng model.
  - Tự động áp dụng batch_size=16, num_workers=0 để chống OOM 6GB VRAM.
  - Tự động lưu log chi tiết từng model vào starter/runs_multi_logs/ .

Cách dùng:
  - Chạy 3 model song song với 3 progress bar cyan:
      python train_multi.py --mode parallel

  - Chạy 3 model nối tiếp lần lượt:
      python train_multi.py --mode sequential

  - Tự chọn danh sách backbone:
      python train_multi.py --models resnet50 convnext_tiny mobilenetv3 --mode parallel
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from tqdm import tqdm

# Đảm bảo mã hóa UTF-8 trên Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STARTER_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

DEFAULT_EXP_CONFIGS = [
    {"exp_id": "B01", "backbone": "resnet50"},
    {"exp_id": "B02", "backbone": "convnext_tiny"},
    {"exp_id": "B03", "backbone": "mobilenetv3"},
]


def run_sequential(configs: list[dict], base_args: list[str]) -> None:
    """Chạy lần lượt từng model theo thứ tự. Tận dụng tối đa 100% GPU cho mỗi model."""
    print("\n" + "=" * 75)
    print(f"[*] BAT DAU CHAY NOI TIEP (SEQUENTIAL) {len(configs)} MO HINH")
    print("=" * 75)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for i, item in enumerate(configs, start=1):
        exp_id = item["exp_id"]
        backbone = item["backbone"]
        print(f"\n>>> [{i}/{len(configs)}] Dang huan luyen {exp_id} ({backbone})...")

        cmd = [
            PYTHON_EXE,
            str(STARTER_DIR / "train.py"),
            "--exp_id", exp_id,
            "--backbone", backbone,
        ] + base_args

        t0 = time.time()
        ret = subprocess.run(cmd, cwd=str(STARTER_DIR), env=env)
        elapsed = time.time() - t0

        if ret.returncode != 0:
            print(f"[!] Canh bao: {exp_id} ({backbone}) ket thuc voi ma loi {ret.returncode}")
        else:
            print(f"[v] Hoan thanh {exp_id} ({backbone}) trong {elapsed:.1f} giay.")


def run_parallel(configs: list[dict], base_args: list[str], epochs: int) -> None:
    """Chạy đồng thời các model với 3 thanh tqdm màu cyan xếp tầng trực tiếp trên màn hình."""
    log_dir = STARTER_DIR / "runs_multi_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"[*] BAT DAU CHAY SONG SONG {len(configs)} MO HINH VOI TQDM CYAN")
    print(f"[*] Log chi tiet tung model luu tai: {log_dir}")
    print("=" * 75 + "\n")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    pbars: dict[str, tqdm] = {}
    processes: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []
    log_files: list = []

    # Khởi tạo thanh tqdm màu cyan cho từng model tại các dòng riêng biệt
    for idx, item in enumerate(configs):
        exp_id = item["exp_id"]
        backbone = item["backbone"]
        pbar = tqdm(
            total=100,
            desc=f"[{exp_id}: {backbone:14s}] Khoi dong... ",
            position=idx,
            leave=True,
            colour="cyan",
            dynamic_ncols=True
        )
        pbars[exp_id] = pbar

    # Hàm đọc log của từng tiến trình và cập nhật thanh progress bar tương ứng
    def worker_reader(exp_id: str, backbone: str, proc: subprocess.Popen, log_file, pbar: tqdm):
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                log_file.write(line)
                log_file.flush()
                raw_line = line.strip()

                if raw_line.startswith("[PBAR]"):
                    # Parse các tham số dạng key=value
                    params = dict(re.findall(r"(\w+)=([\w\.\-\+]+)", raw_line))
                    phase = params.get("phase", "train")

                    if phase == "train":
                        step = int(params.get("step", 0))
                        total = int(params.get("total", 100))
                        ep = int(params.get("epoch", 1))
                        loss_str = params.get("loss", "")
                        lr_str = params.get("lr", "")

                        pbar.total = total
                        pbar.n = step
                        pbar.set_description(f"[{exp_id}: {backbone:14s}] Ep {ep:02d}/{epochs:02d} [Train]")
                        pbar.set_postfix({"loss": loss_str, "lr": lr_str})
                        pbar.refresh()

                    elif phase == "val":
                        step = int(params.get("step", 0))
                        total = int(params.get("total", 100))
                        pbar.total = total
                        pbar.n = step
                        pbar.set_description(f"[{exp_id}: {backbone:14s}] Danh gia [Val]")
                        pbar.refresh()

                    elif phase == "epoch_end":
                        ep = int(params.get("epoch", 1))
                        f1_val = params.get("val_f1", "-")
                        loss_val = params.get("val_loss", "-")
                        tqdm.write(f"=> [{exp_id}: {backbone:14s}] Epoch {ep:02d}/{epochs:02d} | Val Macro-F1: {f1_val} | Val Loss: {loss_val}")

            proc.wait()
            if proc.returncode == 0:
                pbar.set_description(f"[{exp_id}: {backbone:14s}] HOAN THANH [xong]")
                pbar.n = pbar.total
                pbar.refresh()
            else:
                pbar.set_description(f"[{exp_id}: {backbone:14s}] BI LOI (code {proc.returncode})")
                pbar.refresh()
        except Exception as e:
            pbar.set_description(f"[{exp_id}: {backbone:14s}] LOI: {e}")
            pbar.refresh()

    try:
        # Khởi động từng tiến trình con với độ trễ 4s để VRAM không bị sốc
        for item in configs:
            exp_id = item["exp_id"]
            backbone = item["backbone"]
            log_path = log_dir / f"{exp_id}_{backbone}.log"
            f = open(log_path, "w", encoding="utf-8", errors="replace")
            log_files.append(f)

            cmd = [
                PYTHON_EXE,
                "-u",  # Unbuffered binary stdout để nhận step real-time
                str(STARTER_DIR / "train.py"),
                "--exp_id", exp_id,
                "--backbone", backbone,
                "--report_progress",
            ] + base_args

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(STARTER_DIR),
                env=env,
                bufsize=1
            )
            processes.append(proc)

            t = threading.Thread(
                target=worker_reader,
                args=(exp_id, backbone, proc, f, pbars[exp_id]),
                daemon=True
            )
            t.start()
            threads.append(t)
            time.sleep(4)

        # Chờ toàn bộ các thread đọc log hoàn thành
        for t in threads:
            t.join()

    except KeyboardInterrupt:
        print("\n\n[!] Nhan lenh dung (Ctrl+C). Dang dung tat ca tien trinh...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
    finally:
        for f in log_files:
            try:
                f.close()
            except Exception:
                pass
        for pbar in pbars.values():
            pbar.close()

    print("\n" + "=" * 75)
    print("HOAN TAT CAC TIEN TRINH:")
    print("=" * 75)
    for item, proc in zip(configs, processes):
        code = proc.returncode
        status = "THANH CONG" if code == 0 else f"LOI (code {code})"
        print(f" - {item['exp_id']} ({item['backbone']}): {status} | Log: runs_multi_logs/{item['exp_id']}_{item['backbone']}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model trainer voi tqdm cyan cho GPU Laptop 6GB")
    parser.add_argument("--mode", type=str, default="parallel", choices=["parallel", "sequential"],
                        help="Che do chay: 'parallel' (song song) hoac 'sequential' (lan luot)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Danh sach backbone, vi du: --models resnet50 convnext_tiny mobilenetv3")
    parser.add_argument("--epochs", type=int, default=12, help="So epoch (mac dinh: 12)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (mac dinh: 16 khi chay parallel, 32 khi sequential)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="So worker DataLoader (mac dinh: 0 khi chay parallel, 2 khi sequential)")
    parser.add_argument("--seed", type=int, default=0, help="Seed ngau nhien (mac dinh: 0)")

    args = parser.parse_args()

    # Xác định danh sách model
    if args.models:
        configs = []
        for i, m in enumerate(args.models, start=1):
            configs.append({"exp_id": f"B{i:02d}", "backbone": m})
    else:
        configs = DEFAULT_EXP_CONFIGS

    # Cấu hình an toàn cho RTX 3050 6GB
    if args.mode == "parallel":
        batch_size = args.batch_size if args.batch_size is not None else 16
        num_workers = args.num_workers if args.num_workers is not None else 0
    else:
        batch_size = args.batch_size if args.batch_size is not None else 32
        num_workers = args.num_workers if args.num_workers is not None else 2

    base_args = [
        "--epochs", str(args.epochs),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--seed", str(args.seed),
    ]

    print("=" * 75)
    print("THIET LAP TIEN TRINH:")
    print(f"  - Che do: {args.mode.upper()}")
    print(f"  - So luong model: {len(configs)}")
    for c in configs:
        print(f"    * {c['exp_id']}: {c['backbone']}")
    print(f"  - Batch size moi model: {batch_size}")
    print(f"  - Num workers: {num_workers}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Mau thanh tien trinh: CYAN")
    print("=" * 75)

    if args.mode == "parallel":
        run_parallel(configs, base_args, epochs=args.epochs)
    else:
        run_sequential(configs, base_args)


if __name__ == "__main__":
    main()
