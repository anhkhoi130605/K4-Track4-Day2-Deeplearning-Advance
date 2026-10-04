#!/usr/bin/env python3
"""eval.py - công cụ đánh giá chính thức của Lab Day 2 (DeepWeeds).

File này đã hoàn chỉnh: sinh viên KHÔNG sửa công thức chỉ số. Mọi con số trong
results.xlsx và báo cáo phải khớp với kết quả của file này (RUBRIC.md, mục I).
Chỉ phụ thuộc numpy và pandas.

Định nghĩa chỉ số (README.md, mục 2.2)
--------------------------------------
- top-1 accuracy : số ảnh đoán đúng / tổng số ảnh, không trọng số theo lớp.
- macro-F1       : trung bình cộng F1 của 9 lớp (lớp không có dự đoán đúng có F1 = 0).
- balanced acc.  : trung bình recall của các lớp có ảnh trong tập.
- precision / recall / F1 theo lớp, ma trận nhầm lẫn (hàng = nhãn thật).
- ECE            : 15 bin đều theo độ tin cậy (= max softmax); bin m là (b_{m-1}, b_m].
                   ECE = sum_m (n_m / n) * |acc_m - conf_m|
- mean +- std    : qua các seed, std mẫu (ddof = 1).

Định dạng file dự đoán (mỗi seed một file)
------------------------------------------
    Filename, y_true, y_pred, p0, p1, ..., p8
p0..p8 là xác suất softmax, thứ tự theo cột `Label` của labels.csv
(0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives). Tên file chứa `seed<k>`,
ví dụ predictions/F01_seed0_test.csv. Hàm `save_predictions(...)` trong file này ghi
đúng định dạng đó (`from eval import save_predictions`).

Cách dùng
---------
    # 1) Tính chỉ số cho một nhóm file (một cấu hình, nhiều seed)
    python eval.py score --pred "predictions/F01_seed*_test.csv" \
        --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag F01

    # 2) Tự chấm phần I của RUBRIC (chung kết so với mốc)
    python eval.py grade \
        --final    "predictions/F01_seed*_test.csv" \
        --baseline "predictions/T00_seed*_test.csv" \
        --uncal    "predictions/F01_uncal_seed*_test.csv" \
        --final-val "predictions/F01_seed*_val.csv" \
        --latency-p95-ms 41.5 --latency-method proper \
        --test-csv labels/test_subset0.csv --labels labels/labels.csv

`--uncal` (cùng cấu hình nhưng chưa temperature scaling) và `--final-val` (dự đoán
trên val) là tùy chọn; thiếu thì các ý I4(a) hoặc I4(b) ghi là "chưa chấm được".
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Hằng số
# --------------------------------------------------------------------------- #
NUM_CLASSES = 9
ECE_BINS = 15
PROB_SUM_TOL = 1e-3  # xác suất mỗi dòng phải cộng bằng 1 (sai số cho phép)

# Thứ tự lớp theo cột `Label` của labels.csv (lấy từ deepweeds.py của tác giả).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]

# Hai lớp khó nhất và mốc recall (%) từ bài báo gốc (README.md, mục 2.3).
HARD_CLASSES = {"Chinee Apple": 88.5, "Snake Weed": 88.8}

# ---- Ngưỡng chấm RUBRIC mục I (TẠM THỜI: giảng viên có thể chỉnh tại đây) ---- #
ACC_TIERS = [(95.7, 7), (95.1, 6), (94.0, 5), (92.0, 3), (90.0, 1)]  # I1: (acc %, điểm)
I2_MIN_DELTA = 0.01          # I2: Delta macro-F1 tối thiểu để đạt 5 điểm
HARD_FLOORS = [(85.0, 3), (80.0, 2)]  # I3: sàn recall (%) cho cả hai lớp khó
I3_MIN_POINTS = 1            # I3: có báo cáo đầy đủ nhưng dưới 80%
I4_MAX_VAL_TEST_GAP = 0.02   # I4(b): |macro-F1 val - macro-F1 test| tối đa
I5_P95_BUDGET_MS = 100.0     # I5: ngân sách p95 ở batch 1
MIN_SEEDS = 3


# --------------------------------------------------------------------------- #
# Đọc và kiểm tra file dự đoán
# --------------------------------------------------------------------------- #
@dataclass
class Pred:
    path: str
    seed: int | None
    filenames: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray


def parse_seed(path: str) -> int | None:
    m = re.search(r"seed[_-]?(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def read_pred(path: str, num_classes: int = NUM_CLASSES) -> Pred:
    """Đọc một file dự đoán và kiểm tra định dạng. Ném ValueError nếu sai."""
    df = pd.read_csv(path)
    prob_cols = [f"p{i}" for i in range(num_classes)]
    missing = [c for c in ["Filename", "y_true", "y_pred", *prob_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: thiếu cột {missing}")
    extra_p = [c for c in df.columns if re.fullmatch(r"p\d+", c) and c not in prob_cols]
    if extra_p:
        raise ValueError(f"{path}: có cột xác suất ngoài p0..p{num_classes - 1}: {extra_p}")
    if len(df) == 0:
        raise ValueError(f"{path}: file rỗng")
    if df["Filename"].duplicated().any():
        raise ValueError(f"{path}: có Filename bị trùng")

    probs = df[prob_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1 + 1e-6).any():
        raise ValueError(f"{path}: xác suất chứa NaN, âm hoặc lớn hơn 1; hãy lưu softmax, không lưu logit")
    bad_sum = np.abs(probs.sum(1) - 1.0) > PROB_SUM_TOL
    if bad_sum.any():
        raise ValueError(f"{path}: {int(bad_sum.sum())} dòng có tổng xác suất khác 1 "
                         f"(sai số > {PROB_SUM_TOL}); hãy lưu softmax, không lưu logit")

    ints = {}
    for col in ("y_true", "y_pred"):
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(v).all() or (v != np.round(v)).any():
            raise ValueError(f"{path}: cột {col} phải là số nguyên")
        v = v.astype(np.int64)
        if v.min() < 0 or v.max() >= num_classes:
            raise ValueError(f"{path}: cột {col} ngoài khoảng 0..{num_classes - 1}")
        ints[col] = v

    argmax = probs.argmax(1)
    mismatch = int((argmax != ints["y_pred"]).sum())
    if mismatch:
        raise ValueError(f"{path}: {mismatch} dòng có y_pred khác argmax của p0..p{num_classes - 1}")

    return Pred(path, parse_seed(path), df["Filename"].to_numpy(), ints["y_true"], ints["y_pred"], probs)


def save_predictions(path: str | Path, filenames, y_true, probs, num_classes: int = NUM_CLASSES) -> Path:
    """Ghi một file dự đoán đúng định dạng của eval.py: Filename, y_true, y_pred, p0..p{K-1}.

    Dùng hàm này (hoặc ghi y hệt định dạng) để tạo predictions/<exp_id>_seed<k>_<split>.csv.
    `probs` là XÁC SUẤT softmax (mỗi dòng cộng bằng 1), không phải logit.
    y_pred được tính bằng argmax của probs. Kết quả luôn qua được `read_pred`.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    filenames = list(filenames)
    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise ValueError(f"probs phải có dạng (N, {num_classes}), nhận {probs.shape}")
    if not (len(filenames) == len(y_true) == len(probs)):
        raise ValueError("filenames, y_true, probs phải cùng độ dài")
    if np.abs(probs.sum(1) - 1.0).max() > PROB_SUM_TOL:
        raise ValueError("probs chưa chuẩn hoá (tổng mỗi dòng khác 1); hãy áp dụng softmax trước khi lưu")
    df = pd.DataFrame({"Filename": filenames, "y_true": y_true, "y_pred": probs.argmax(1)})
    for i in range(num_classes):
        df[f"p{i}"] = probs[:, i]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.8g")
    return path


def check_against_csv(pred: Pred, csv_path: str, what: str = "test") -> None:
    """Đối chiếu tên file và nhãn thật với file CSV chia sẵn (Filename, Label)."""
    ref = pd.read_csv(csv_path)
    if not {"Filename", "Label"} <= set(ref.columns):
        raise ValueError(f"{csv_path}: cần cột Filename và Label")
    ref_map = dict(zip(ref["Filename"], ref["Label"].astype(int)))
    names = set(pred.filenames)
    if names != set(ref_map):
        only_pred = len(names - set(ref_map))
        only_ref = len(set(ref_map) - names)
        raise ValueError(f"{pred.path}: không khớp {what} CSV ({csv_path}): "
                         f"thừa {only_pred} ảnh, thiếu {only_ref} ảnh")
    wrong = sum(int(ref_map[f]) != int(t) for f, t in zip(pred.filenames, pred.y_true))
    if wrong:
        raise ValueError(f"{pred.path}: {wrong} dòng có y_true khác Label trong {csv_path}")


def load_names(labels_csv: str | None) -> list[str]:
    """Lấy tên lớp từ labels.csv (cột Label, Species) nếu có; nếu không dùng mặc định."""
    if not labels_csv:
        return list(CLASS_NAMES)
    df = pd.read_csv(labels_csv)
    if not {"Label", "Species"} <= set(df.columns):
        raise ValueError(f"{labels_csv}: cần cột Label và Species")
    m = df.drop_duplicates("Label").sort_values("Label")
    if list(m["Label"]) != list(range(NUM_CLASSES)):
        raise ValueError(f"{labels_csv}: Label phải là 0..{NUM_CLASSES - 1}")
    return [str(s) for s in m["Species"]]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def hard_class_indices(names: list[str]) -> dict[str, int]:
    out = {}
    for hard in HARD_CLASSES:
        for i, n in enumerate(names):
            if _norm(n) == _norm(hard):
                out[hard] = i
    missing = set(HARD_CLASSES) - set(out)
    if missing:
        raise ValueError(f"không tìm thấy lớp {sorted(missing)} trong danh sách tên lớp {names}")
    return out


def expand(pattern: str | list[str]) -> list[str]:
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"không có file nào khớp: {patterns}")
    return files


# --------------------------------------------------------------------------- #
# Chỉ số
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    predicted = cm.sum(0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def ece_score(probs: np.ndarray, y_true: np.ndarray, bins: int = ECE_BINS) -> float:
    """ECE với `bins` bin đều; độ tin cậy = max softmax; bin m là (b_{m-1}, b_m]."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(np.float64)
    idx = np.clip(np.ceil(conf * bins).astype(np.int64) - 1, 0, bins - 1)
    n = len(conf)
    ece = 0.0
    for m in range(bins):
        mask = idx == m
        if mask.any():
            ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                    k: int = NUM_CLASSES, bins: int = ECE_BINS) -> dict:
    cm = confusion_matrix(y_true, y_pred, k)
    pc = per_class(cm)
    has_support = pc["support"] > 0
    nll = float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, None)).mean())
    return {
        "n": int(len(y_true)),
        "top1": float((y_true == y_pred).mean()),
        "macro_f1": float(pc["f1"].mean()),
        "balanced_acc": float(pc["recall"][has_support].mean()),
        "ece": ece_score(probs, y_true, bins),
        "nll": nll,
        "precision": pc["precision"], "recall": pc["recall"], "f1": pc["f1"],
        "support": pc["support"], "confusion": cm,
    }


SCALARS = ("top1", "macro_f1", "balanced_acc", "ece", "nll")
VECTORS = ("precision", "recall", "f1")


def mean_std(values) -> tuple[float, float]:
    """mean và std mẫu (ddof=1) theo trục seed; std là NaN nếu chỉ có 1 seed."""
    a = np.asarray(values, dtype=np.float64)
    mean = a.mean(axis=0)
    std = a.std(axis=0, ddof=1) if len(a) > 1 else np.full(a.shape[1:], np.nan)
    if a.ndim == 1:
        return float(mean), float(std)
    return mean, std


@dataclass
class Group:
    preds: list[Pred]
    metrics: list[dict]
    summary: dict  # {"top1": (mean, std), ..., "recall": (vec_mean, vec_std), ...}

    @property
    def seeds(self) -> list[int | None]:
        return [p.seed for p in self.preds]


def load_group(pattern, ref_csv: str | None, k: int = NUM_CLASSES, ref_what: str = "test") -> Group:
    files = expand(pattern)
    preds = [read_pred(f, k) for f in files]
    if ref_csv:
        for p in preds:
            check_against_csv(p, ref_csv, ref_what)
    base_names = set(preds[0].filenames)
    for p in preds[1:]:
        if set(p.filenames) != base_names:
            raise ValueError(f"{p.path}: tập Filename khác {preds[0].path}")
    seeds = [p.seed for p in preds if p.seed is not None]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"seed bị trùng giữa các file: {files}")
    metrics = [compute_metrics(p.y_true, p.y_pred, p.probs, k) for p in preds]
    summary = {key: mean_std([m[key] for m in metrics]) for key in (*SCALARS, *VECTORS)}
    return Group(preds, metrics, summary)


# --------------------------------------------------------------------------- #
# Định dạng đầu ra
# --------------------------------------------------------------------------- #
def fmt(mean: float, std: float, digits: int = 4) -> str:
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f} (1 seed)"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def report_group(name: str, g: Group, names: list[str]) -> str:
    lines = [f"### {name} ({len(g.preds)} seed: {g.seeds}; {g.metrics[0]['n']} ảnh)", ""]
    if len(g.preds) < MIN_SEEDS:
        lines += [f"> CẢNH BÁO: chỉ có {len(g.preds)} seed, yêu cầu tối thiểu {MIN_SEEDS}.", ""]
    lines += ["| Chỉ số | mean ± std |", "|---|---|"]
    labels = {"top1": "top-1 accuracy", "macro_f1": "macro-F1", "balanced_acc": "balanced accuracy",
              "ece": "ECE (15 bin)", "nll": "NLL"}
    for key in SCALARS:
        lines.append(f"| {labels[key]} | {fmt(*g.summary[key])} |")
    lines += ["", "| Lớp | Precision | Recall | F1 | Số ảnh |", "|---|---|---|---|---|"]
    pm, ps = g.summary["precision"]
    rm, rs = g.summary["recall"]
    fm, fs = g.summary["f1"]
    for i, n in enumerate(names):
        lines.append(f"| {n} | {fmt(pm[i], ps[i], 3)} | {fmt(rm[i], rs[i], 3)} | "
                     f"{fmt(fm[i], fs[i], 3)} | {int(g.metrics[0]['support'][i])} |")
    return "\n".join(lines)


def save_group(out_dir: Path, tag: str, g: Group, names: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p, m in zip(g.preds, g.metrics):
        rows.append({"file": Path(p.path).name, "seed": p.seed,
                     **{k: m[k] for k in ("n", *SCALARS)}})
    pd.DataFrame(rows).to_csv(out_dir / f"{tag}_per_seed.csv", index=False)
    pc = pd.DataFrame({"class": names})
    for key in VECTORS:
        mean, std = g.summary[key]
        pc[f"{key}_mean"], pc[f"{key}_std"] = mean, std
    pc["support"] = g.metrics[0]["support"].astype(int)
    pc.to_csv(out_dir / f"{tag}_per_class.csv", index=False)
    cm = sum(m["confusion"] for m in g.metrics)
    pd.DataFrame(cm, index=[f"true_{n}" for n in names],
                 columns=[f"pred_{n}" for n in names]).to_csv(out_dir / f"{tag}_confusion_sum.csv")
    summary = {"seeds": g.seeds,
               **{k: {"mean": g.summary[k][0], "std": g.summary[k][1]} for k in SCALARS},
               **{k: {"mean": g.summary[k][0].tolist(), "std": g.summary[k][1].tolist()} for k in VECTORS},
               "classes": names}
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Lệnh `score`
# --------------------------------------------------------------------------- #
def cmd_score(args) -> int:
    names = load_names(args.labels)
    g = load_group(args.pred, args.test_csv, ref_what="test")
    if not args.test_csv:
        print("LƯU Ý: chưa đối chiếu với test CSV (thêm --test-csv labels/test_subset0.csv)\n")
    print(report_group(args.tag, g, names))
    if args.out:
        save_group(Path(args.out), args.tag, g, names)
        print(f"\nĐã lưu kết quả vào {args.out}/{args.tag}_*")
    return 0


# --------------------------------------------------------------------------- #
# Lệnh `grade` (RUBRIC mục I)
# --------------------------------------------------------------------------- #
def points_i1(acc_pct: float) -> int:
    for thr, pts in ACC_TIERS:
        if acc_pct >= thr:
            return pts
    return 0


def points_i2(delta: float, s: float) -> int:
    if not math.isfinite(delta) or delta <= 0:
        return 0
    s = 0.0 if not math.isfinite(s) else s
    if delta > s and delta >= I2_MIN_DELTA:
        return 5
    return 4 if delta > s else 2


def points_i3(recalls_pct: dict[str, float]) -> int:
    if all(recalls_pct[c] >= ref for c, ref in HARD_CLASSES.items()):
        return 4
    low = min(recalls_pct.values())
    for floor, pts in HARD_FLOORS:
        if low >= floor:
            return pts
    return I3_MIN_POINTS


def cmd_grade(args) -> int:
    names = load_names(args.labels)
    hard_idx = hard_class_indices(names)
    final = load_group(args.final, args.test_csv, ref_what="test")
    base = load_group(args.baseline, args.test_csv, ref_what="test")
    if not args.test_csv:
        print("LƯU Ý: chưa đối chiếu với test CSV (thêm --test-csv labels/test_subset0.csv)\n")
    warnings: list[str] = []
    for label, g in (("chung kết", final), ("mốc", base)):
        if len(g.preds) < MIN_SEEDS:
            warnings.append(f"nhóm {label} chỉ có {len(g.preds)} seed (< {MIN_SEEDS})")
    if set(final.preds[0].filenames) != set(base.preds[0].filenames):
        warnings.append("tập Filename của chung kết và mốc khác nhau")

    items: list[tuple[str, str, int | None, int, str]] = []  # (mã, tiêu chí, điểm, tối đa, ghi chú)

    # I1
    acc = final.summary["top1"][0] * 100
    items.append(("I1", "Top-1 accuracy test", points_i1(acc), 7, f"{acc:.2f}% (mean {len(final.preds)} seed)"))

    # I2
    mf_f, sd_f = final.summary["macro_f1"]
    mf_b, sd_b = base.summary["macro_f1"]
    delta = mf_f - mf_b
    s = max([v for v in (sd_f, sd_b) if math.isfinite(v)], default=math.nan)
    items.append(("I2", "Macro-F1 cải thiện so với mốc", points_i2(delta, s), 5,
                  f"final {mf_f:.4f}, mốc {mf_b:.4f}, Δ={delta:+.4f}, s={s:.4f}"))

    # I3
    rec_mean = final.summary["recall"][0] * 100
    recalls = {c: float(rec_mean[i]) for c, i in hard_idx.items()}
    note = ", ".join(f"{c} {recalls[c]:.1f}% (mốc {HARD_CLASSES[c]}%)" for c in HARD_CLASSES)
    items.append(("I3", "Recall hai lớp khó", points_i3(recalls), 4, note))

    # I4(a)
    if args.uncal:
        uncal = load_group(args.uncal, args.test_csv, ref_what="test")
        e_after, e_before = final.summary["ece"][0], uncal.summary["ece"][0]
        items.append(("I4a", "ECE sau TS < ECE trước", int(e_after < e_before), 1,
                      f"trước {e_before:.4f}, sau {e_after:.4f}"))
    else:
        items.append(("I4a", "ECE sau TS < ECE trước", None, 1, "chưa chấm được (thiếu --uncal)"))

    # I4(b)
    if args.final_val:
        fv = load_group(args.final_val, args.val_csv, ref_what="val")
        gap = abs(fv.summary["macro_f1"][0] - mf_f)
        items.append(("I4b", "Chênh macro-F1 val/test <= %.2f" % I4_MAX_VAL_TEST_GAP,
                      int(gap <= I4_MAX_VAL_TEST_GAP), 1,
                      f"val {fv.summary['macro_f1'][0]:.4f}, test {mf_f:.4f}, chênh {gap:.4f}"))
    else:
        items.append(("I4b", "Chênh macro-F1 val/test <= %.2f" % I4_MAX_VAL_TEST_GAP, None, 1,
                      "chưa chấm được (thiếu --final-val)"))

    # I5
    if args.latency_p95_ms is None:
        items.append(("I5", "Cấu hình thời gian thực", None, 2, "chưa chấm được (thiếu --latency-p95-ms)"))
    else:
        ok = args.latency_p95_ms <= I5_P95_BUDGET_MS
        pts = 0 if not ok else (2 if args.latency_method == "proper" else 1)
        items.append(("I5", "Cấu hình thời gian thực", pts, 2,
                      f"p95 = {args.latency_p95_ms:.1f} ms (ngân sách {I5_P95_BUDGET_MS:.0f} ms), "
                      f"đo {'đúng cách' if args.latency_method == 'proper' else 'chưa đủ warmup/synchronize'}"))

    scored = [i for i in items if i[2] is not None]
    got = sum(i[2] for i in scored)
    max_scored = sum(i[3] for i in scored)
    lines = ["## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)", "",
             "| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |", "|---|---|---|---|---|"]
    for code, crit, pts, mx, note in items:
        lines.append(f"| {code} | {crit} | {'-' if pts is None else pts} | {mx} | {note} |")
    lines += ["", f"**Tổng các ý đã chấm: {got} / {max_scored}** (phần I tối đa 20)."]
    pending = [i[0] for i in items if i[2] is None]
    if pending:
        lines.append(f"Chưa chấm được: {', '.join(pending)}.")
    if warnings:
        lines += ["", "Cảnh báo:"] + [f"- {w}" for w in warnings]
    lines += ["", "Ngưỡng điểm là TẠM THỜI (xem khối hằng số đầu file eval.py và RUBRIC.md mục I)."]
    print("\n".join(lines))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "grade_I.json").write_text(json.dumps({
            "items": [{"code": c, "criterion": cr, "points": p, "max": m, "note": n} for c, cr, p, m, n in items],
            "total": got, "max_scored": max_scored, "warnings": warnings}, indent=2, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Đánh giá Lab Day 2 (DeepWeeds)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="tính chỉ số cho một nhóm file dự đoán (nhiều seed)")
    sc.add_argument("--pred", nargs="+", required=True, help="glob của file dự đoán")
    sc.add_argument("--test-csv", help="labels/test_subset0.csv để đối chiếu tên file và nhãn")
    sc.add_argument("--labels", help="labels/labels.csv để lấy tên lớp")
    sc.add_argument("--tag", default="score")
    sc.add_argument("--out", help="thư mục lưu JSON/CSV")
    sc.set_defaults(func=cmd_score)

    gr = sub.add_parser("grade", help="tự chấm phần I của RUBRIC")
    gr.add_argument("--final", nargs="+", required=True, help="glob dự đoán test của cấu hình chung kết")
    gr.add_argument("--baseline", nargs="+", required=True, help="glob dự đoán test của mốc (T00 + I00)")
    gr.add_argument("--uncal", nargs="+", help="glob dự đoán test của chung kết khi chưa temperature scaling")
    gr.add_argument("--final-val", nargs="+", help="glob dự đoán val của chung kết")
    gr.add_argument("--latency-p95-ms", type=float, help="p95 batch 1 của cấu hình thời gian thực (ms)")
    gr.add_argument("--latency-method", choices=["proper", "partial"], default="proper",
                    help="proper = có warmup và synchronize; partial = chưa đủ")
    gr.add_argument("--test-csv", help="labels/test_subset0.csv")
    gr.add_argument("--val-csv", help="labels/val_subset0.csv (đối chiếu --final-val)")
    gr.add_argument("--labels", help="labels/labels.csv")
    gr.add_argument("--out", help="thư mục lưu grade_I.json")
    gr.set_defaults(func=cmd_grade)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
