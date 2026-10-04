"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path
import sys

# Dam bao terminal Windows / pipe khong bi loi charmap UnicodeEncodeError
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

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, ToPILImage, ToTensor, Normalize, Resize, RandomAffine, ColorJitter
import pandas as pd

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR / "data"
LABELS_DIR = ROOT_DIR / "labels"


def _resolve_labels_dir(labels_dir: str | Path) -> Path:
    """Chấp nhận đường dẫn đến folder labels hoặc project root; tự dò vị trí đúng."""
    labels_dir = Path(labels_dir).expanduser()
    if labels_dir.exists():
        return labels_dir

    candidates = [
        CURRENT_DIR / "data" / "labels",
        CURRENT_DIR / "labels",
        CURRENT_DIR.parent / "labels",
        CURRENT_DIR.parent / "starter" / "data" / "labels",
        CURRENT_DIR.parent / "starter" / "labels",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy folder labels từ {labels_dir}. Kiểm tra bạn đang ở đúng repo/submission.")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

        Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
        KHÔNG sửa, lọc hay chia lại dữ liệu.

        TODO:
          - đọc ba file CSV bằng pandas
          - trả về (train_df, val_df, test_df)
        """
    labels_path = _resolve_labels_dir(labels_dir)

    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    assert train_file.exists(), f"Không tìm thấy file: {train_file}"
    assert val_file.exists(), f"Không tìm thấy file: {val_file}"
    assert test_file.exists(), f"Không tìm thấy file: {test_file}"

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu."""
    images_dir = Path(images_dir)
    assert images_dir.exists() and images_dir.is_dir(), f"Folder ảnh {images_dir} không tồn tại."

    # Lấy danh sách tên file theo cột 'Filename'
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_imgs = n_train + n_val + n_test

    # --- 1. Tỉ lệ xấp xỉ 60/20/20 & số lượng mỗi lớp ---
    ratio_train = n_train / total_imgs if total_imgs > 0 else 0
    ratio_val = n_val / total_imgs if total_imgs > 0 else 0
    ratio_test = n_test / total_imgs if total_imgs > 0 else 0

    print("--- THONG KE SO LUONG VA TI LE ---")
    print(f"Train: {n_train} ({ratio_train:.2%})")
    print(f"Val:   {n_val} ({ratio_val:.2%})")
    print(f"Test:  {n_test} ({ratio_test:.2%})")
    print(f"Tong:  {total_imgs}")

    assert 0.55 <= ratio_train <= 0.65, f"Tỉ lệ tập Train không đạt ~60%: {ratio_train:.2%}"
    assert 0.15 <= ratio_val <= 0.25, f"Tỉ lệ tập Val không đạt ~20%: {ratio_val:.2%}"
    assert 0.15 <= ratio_test <= 0.25, f"Tỉ lệ tập Test không đạt ~20%: {ratio_test:.2%}"

    label_col = "Label" if "Label" in train_df.columns else "Species"
    per_class = {
            "train": train_df[label_col].value_counts().to_dict(),
            "val": val_df[label_col].value_counts().to_dict(),
            "test": test_df[label_col].value_counts().to_dict(),
        }

    # --- 2. Giao từng cặp tập theo Filename phải RỖNG ---
    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files

    assert len(train_val_overlap) == 0, f"Rò rỉ dữ liệu: Train và Val trùng {len(train_val_overlap)} ảnh!"
    assert len(train_test_overlap) == 0, f"Rò rỉ dữ liệu: Train và Test trùng {len(train_test_overlap)} ảnh!"
    assert len(val_test_overlap) == 0, f"Rò rỉ dữ liệu: Val và Test trùng {len(val_test_overlap)} ảnh!"

    # --- 3. Hợp ba tập phải bằng đúng 17.509 ảnh ---
    all_files = train_files | val_files | test_files
    EXPECTED_TOTAL = 17509
    assert len(all_files) == EXPECTED_TOTAL, (
            f"Tổng số ảnh duy nhất ({len(all_files)}) không đúng kỳ vọng ({EXPECTED_TOTAL})!"
        )
    assert total_imgs == EXPECTED_TOTAL, (
            f"Tổng số dòng ({total_imgs}) không đúng kỳ vọng ({EXPECTED_TOTAL})!"
        )

    # --- 4. Mọi Filename đều tồn tại trong images_dir ---
    missing_files = [f for f in all_files if not (images_dir / f).is_file()]
    assert len(missing_files) == 0, (
            f"Phát hiện {len(missing_files)} file không tồn tại trong {images_dir}! "
            f"Ví dụ: {missing_files[:5]}"
        )

    print("\nTat ca cac kiem tra split deu HOP LE!")

    return {
            "n": {
                "train": n_train,
                "val": n_val,
                "test": n_test,
                "total": total_imgs,
                "ratios": {
                    "train": round(ratio_train, 4),
                    "val": round(ratio_val, 4),
                    "test": round(ratio_test, 4),
                },
            },
            "per_class": per_class,
            "overlap": {
                "train_val": len(train_val_overlap),
                "train_test": len(train_test_overlap),
                "val_test": len(val_test_overlap),
            },
            "all_files_exist": True,
        }
def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    TODO: dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    if not train:
        return Compose([
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        # 2. Nếu là tập Train (có Augmentation)
        # Cơ bản: Resize về img_size + xoay nhẹ / lật + ToTensor + Normalize
    train_transforms = [
        Resize((img_size, img_size)),
        RandomAffine(
            degrees=(-10, 10),
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
        ),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        ),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return Compose(train_transforms)
class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    TODO:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        filename = self.filenames[i]
        label = self.labels[i]
        img_path = self.images_dir / filename

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label, filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 1):
    """Tạo DataLoader.

    TODO:
    - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
      (thứ tự phải ổn định để ghép logit với Filename)
    - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
      1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
    - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """

    #1. Khởi tạo Dataset
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    sampler_obj = None
    shuffle = False

    if train:
        if sampler == "balanced":
            # Trọng số cân bằng mẫu: 1 / (số lượng ảnh của lớp đó)
            class_counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / class_counts[label] for label in df["Label"]]
            sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
            sampler_obj = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        else:
            shuffle = True
    else:
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
    return loader

if __name__ == "__main__":
    labels_dir = Path(__file__).resolve().parent / "data" / "labels"
    images_dir = Path(__file__).resolve().parent / "data" / "images"
    train_df, val_df, test_df = load_split(labels_dir, fold=0)
    # print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    # report = check_split(train_df, val_df, test_df, images_dir)
    # print(report)



    # print("=== train_df.head() ===")
    # print(train_df.head())
    # print()
    #
    # print("=== val_df.head() ===")
    # print(val_df.head())
    # print()
    #
    # print("=== test_df.head() ===")
    # print(test_df.head())
    # print()
    #
    # print("=== số lượng từng tập ===")
    # print("train:", len(train_df))
    # print("val:", len(val_df))
    # print("test:", len(test_df))
    # print()
    #
    # print("=== kiểm tra split ===")
    # report = check_split(train_df, val_df, test_df, images_dir)
    # print(report)




    files = list(images_dir.iterdir())
    # print("Đường dẫn file:", files[0])
    # # 2. Đọc file ảnh vào bộ nhớ bằng PIL
    # img = Image.open(files[0])
    # print("Kích thước ảnh gốc (Width, Height):", img.size)
    # print("Hệ màu:", img.mode)  # Kết quả: RGB
    # # 3. Chuyển sang mảng numpy để xem dtype và shape
    # arr = np.array(img)
    # print("Kiểu dữ liệu (dtype):", arr.dtype)
    # print("Shape (H, W, C):", arr.shape)
    # Kích thước ảnh gốc (Width, Height): (256, 256)
    # Hệ màu: RGB
    # Kiểu dữ liệu (dtype): uint8
    # Shape (H, W, C): (256, 256, 3)

    # for f in files[:5]:
    #     img = Image.open(f)
    #     plt.imshow(img)
    #     plt.title(f.name)
    #     plt.axis("off")
    #     plt.show()

    train_transform = build_transforms(train=True, img_size=224, aug="basic")
    
    # Lấy ngẫu nhiên 5 ảnh từ tập train (mỗi lần chạy sẽ ra 5 ảnh khác nhau)
    sample_df = train_df.sample(n=5).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("KIỂM TRA NGẪU NHIÊN 5 ẢNH TỪ TẬP TRAIN:")
    print("=" * 70)

    # 3. Chuẩn bị vẽ 5 ảnh cạnh nhau (1 hàng, 5 cột)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i in range(5):
        row = sample_df.iloc[i]
        filename = row["Filename"]
        label = int(row["Label"])
        species_name = CLASS_NAMES[label]
        img_path = images_dir / filename
        # Đọc ảnh gốc bằng PIL
        img_raw = Image.open(img_path)
        # Chạy qua pipeline transform (Resize, Augmentation, ToTensor, Normalize)
        img_tensor = train_transform(img_raw)
        # In thông tin kiểm tra ra terminal
        print(f"[{i + 1}] File: {filename}")
        print(f"    - Nhãn (Label): {label} ({species_name})")
        print(f"    - Kích thước ảnh gốc (W, H): {img_raw.size}")
        print(f"    - Tensor Shape (C, H, W): {tuple(img_tensor.shape)} | Dtype: {img_tensor.dtype}")
        print(f"    - Giá trị Pixel Tensor: Min = {img_tensor.min():.2f}, Max = {img_tensor.max():.2f}")
        print("-" * 50)
        # Hiển thị ảnh gốc lên đồ thị
        axes[i].imshow(img_raw)
        axes[i].set_title(f"Ảnh {i + 1}: {species_name}\n(Nhãn: {label})\n{filename}", fontsize=9)
        axes[i].axis("off")
    plt.suptitle("5 Ảnh mẫu ngẫu nhiên từ tập Train (DeepWeeds)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()