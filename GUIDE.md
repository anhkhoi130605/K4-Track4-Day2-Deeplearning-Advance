# GUIDE — Quy trình thực hiện Lab Day 2

Hướng dẫn này đi theo thứ tự bạn nên làm. Hãy đọc hết một lượt trước khi bắt đầu. Các số liệu và trích dẫn từ slide Day 2 ghi kèm số trang; mọi mốc thời gian GPU trong tài liệu này là **ước lượng chưa được đo**, hãy tự đo epoch đầu tiên rồi tính lại.

## Mục lục

0. [Nguyên tắc thực nghiệm (đọc kỹ)](#0-nguyên-tắc-thực-nghiệm)
1. [Bước 0: Chuẩn bị, EDA, dựng pipeline](#1-bước-0-chuẩn-bị-eda-dựng-pipeline)
2. [Bước 1: So sánh backbone](#2-bước-1-so-sánh-backbone--5)
3. [Bước 2: Công thức huấn luyện](#3-bước-2-công-thức-huấn-luyện)
4. [Bước 3: Phương pháp suy luận](#4-bước-3-phương-pháp-suy-luận)
5. [Bước 4: Cấu hình tốt nhất và chạy test](#5-bước-4-cấu-hình-tốt-nhất-và-chạy-test)
6. [Bước 5: Sản phẩm (xlsx, biểu đồ, báo cáo)](#6-bước-5-sản-phẩm)
7. [Ngân sách tính toán và mẹo Colab/Kaggle](#7-ngân-sách-tính-toán-và-mẹo-colabkaggle)
8. [Bẫy thường gặp](#8-bẫy-thường-gặp)
9. [Câu hỏi gợi ý cho phần phân tích](#9-câu-hỏi-gợi-ý-cho-phần-phân-tích)

---

## 0. Nguyên tắc thực nghiệm

Bài lab chấm **độ chặt chẽ của thí nghiệm** ngang với kết quả cuối. Năm nguyên tắc sau áp dụng cho mọi bước.

**N1. Một thay đổi mỗi lần.** Khi so sánh hai cấu hình, chỉ khác nhau đúng một yếu tố, mọi thứ còn lại giữ nguyên (cùng backbone, cùng số epoch, cùng seed khởi đầu, cùng split). Nếu đổi hai thứ cùng lúc, bạn không kết luận được yếu tố nào có tác dụng.

**N2. Chia dữ liệu cố định.** Dùng `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`. Quy tắc đầy đủ (S1–S6) và các kiểm tra bắt buộc nằm ở [`README.md` mục 2.1](README.md#21-quy-tắc-chia-train--val--test-bắt-buộc); đọc kỹ trước khi làm.

| Tập | Dùng để |
|---|---|
| train | Cập nhật trọng số |
| val | Chọn backbone, siêu tham số, phương pháp suy luận, early stopping, khớp nhiệt độ T |
| test | **Chỉ dùng một lần ở Bước 4** để báo cáo con số cuối cùng |

Nếu bạn nhìn điểm test rồi quay lại chỉnh cấu hình, con số test không còn đáng tin và sẽ bị trừ điểm (xem `RUBRIC.md`). Bạn được lưu logit test vào file khi chạy, nhưng chưa mở hay tính chỉ số trên chúng cho tới Bước 4.

**N3. Chỉ số.**

- **Chỉ số chính: macro-F1** trên val. Dataset mất cân bằng nên accuracy bị lớp `Negative` kéo cao và che lỗi ở các lớp hiếm.
- Chỉ số phụ: top-1 accuracy, balanced accuracy, F1 theo từng lớp, ECE (ở Bước 3).
- Luôn ghi cả tên chỉ số lẫn tập đo (val hay test).
- Định nghĩa chính xác từng chỉ số (macro-F1 trên 9 lớp, ECE 15 bin, mean ± std với `ddof=1`...) và quy trình đánh giá nằm ở [`README.md` mục 2.2](README.md#22-cách-đánh-giá-bắt-buộc). Mọi con số của bạn phải theo định nghĩa đó.

**N4. Seed và nhiễu.** Slide (trang 59) cho biết ResNet-50 lệch chuẩn khoảng 0,10 điểm qua 100 seed, nên chênh lệch dưới ~0,3 điểm là nhiễu. Trên tập nhỏ như DeepWeeds, độ lệch có thể lớn hơn, hãy tự đo. Quy tắc:

- Quét sàng (Bước 1 và 2): 1 seed là chấp nhận được, nhưng chỉ kết luận "tốt hơn" khi chênh lệch **rõ ràng**, và phải ghi chú là mới 1 seed.
- Các cấu hình lọt vào **vòng chung kết** (Bước 4): **≥ 3 seed**, báo cáo mean ± std.
- Khi chênh lệch giữa hai cấu hình nhỏ hơn std, hãy viết "không phân biệt được", không viết "A tốt hơn B".

**N5. Ghi lại tất cả.** Mỗi lần chạy phải lưu: cấu hình đầy đủ (hyperparameter, seed, version thư viện), log theo epoch, đường cong, logit dự đoán trên val và test, thời gian train, checkpoint tốt nhất. Đặt tên theo `exp_id` (xem mục 6.1).

---

## 1. Bước 0: Chuẩn bị, EDA, dựng pipeline

### 1.1 Tải dữ liệu

Tải `images.zip` từ Zenodo, kiểm tra MD5, giải nén; tải các file CSV trong `labels/` từ GitHub (xem README mục 2). Với 17,5k ảnh nhỏ (256×256), bạn có thể giải nén vào đĩa cục bộ của notebook cho nhanh, tránh đọc từng ảnh từ Drive.

### 1.2 EDA (bắt buộc, đưa vào báo cáo)

- Đếm số ảnh theo lớp trong train, val, test của fold 0 và chạy các kiểm tra ở [`README.md` mục 2.1](README.md#21-quy-tắc-chia-train--val--test-bắt-buộc) (giao rỗng, hợp đủ 17.509 ảnh, file tồn tại). Vẽ biểu đồ cột. Tỉ lệ lớp lớn nhất so với lớp nhỏ nhất là bao nhiêu? Đối chiếu với Table 1 của bài báo (`Negative` 9.106; các loài 1.009–1.125 ảnh).
- Xem trực quan ít nhất 3 ảnh mỗi lớp. Lớp nào dễ nhầm với nhau bằng mắt thường? `Negative` trông như thế nào?
- Kiểm tra không trùng ảnh giữa train, val, test (theo tên file).
- Nhìn thống kê ảnh (kích thước, kênh) để chọn chuẩn hoá đầu vào.

Slide (trang 59) khuyên **"nhìn dữ liệu"** trước mọi bước khác.

### 1.3 Dựng pipeline và kiểm tra tính đúng đắn

Làm đúng các bước kiểm tra trong checklist gỡ lỗi của slide (trang 59) **trước khi chạy thí nghiệm thật**:

1. Cố định seed (`random`, `numpy`, `torch`, và DataLoader worker).
2. **Mất mát ban đầu:** với 9 lớp, loss CE ban đầu của một mạng khởi tạo hợp lý phải xấp xỉ `-ln(1/9) ≈ 2,197`. Nếu lệch xa, kiểm tra lại (đặc biệt khi dùng head mới).
3. **Quá khớp một batch nhỏ** (vài mẫu) tới loss gần 0. Nếu không làm được thì lỗi nằm ở code hoặc model, chưa nên chạy tiếp.
4. Kiểm tra đầu vào sau tiền xử lý: vẽ vài ảnh sau augmentation (đã giải chuẩn hoá) để chắc chắn ảnh và nhãn khớp nhau.
5. Kiểm tra chế độ `model.train()` và `model.eval()` được dùng đúng lúc (BatchNorm, dropout).

### 1.4 Công thức nền (baseline recipe)

Từ Bước 1 trở đi, mọi backbone đều dùng **cùng một công thức nền** để so sánh công bằng. Đề xuất, dựa trên ví dụ nhóm tham số ở slide (trang 52):

| Thành phần | Giá trị đề xuất |
|---|---|
| Khởi tạo | Trọng số tiền huấn luyện ImageNet, thay head mới 9 lớp, tinh chỉnh toàn bộ |
| Đầu vào | Train: `RandomResizedCrop(224)` + lật ngang. Val/test: resize 256 rồi `CenterCrop(224)` hoặc dùng thẳng 256 (ghi rõ bạn chọn gì) |
| Chuẩn hoá | Mean/std của ImageNet (hoặc đúng cấu hình của trọng số bạn dùng) |
| Optimizer | AdamW |
| LR | Backbone `1e-4`, head mới `1e-3` (gấp 10 lần) |
| Weight decay | `0,05`, **không áp dụng cho norm và bias** (`weight_decay = 0`) |
| Lịch LR | Warmup khoảng 1 epoch, sau đó cosine về gần 0 |
| Loss | Cross-entropy |
| Batch size | 64 (giảm nếu hết bộ nhớ, và ghi lại) |
| Epoch | 10–15, giống nhau cho mọi backbone |
| Mixed precision | Bật (AMP) |
| Chọn checkpoint | Epoch có macro-F1 val cao nhất |

Đây là điểm xuất phát, không phải chân lý. Với ViT/Swin, LR có thể cần nhỏ hơn. Nếu bạn đổi công thức nền, hãy đổi **cho tất cả** backbone và ghi lại lý do.

### 1.5 Dùng bộ khung `starter/` và `eval.py`

Chi tiết từng file ở [`README.md` mục 2.4](README.md#24-công-cụ-evalpy-và-bộ-khung-starter). Quy trình gợi ý:

1. Chép `starter/` thành `code/` trong thư mục bài nộp. Mở `lab_day2.ipynb` trên Colab hoặc Kaggle, chạy ô cài đặt và tải dữ liệu.
2. Hoàn thiện theo thứ tự: `dataset.py` → `model.py` → `losses.py` → `train.py`. Sau mỗi file, tự viết một kiểm tra nhỏ (ví dụ focal loss với `γ = 0` phải bằng CE; một batch phải overfit được). **Dừng lại ở bước kiểm tra pipeline (mục 1.3) trước khi chạy thí nghiệm thật.**
3. Mọi thí nghiệm đi qua **một** hàm `train.run(Config(...))`; đổi thí nghiệm bằng cách đổi `Config`, không copy code.
4. Khi cần ghi dự đoán, dùng nguyên `eval.save_predictions(...)` (`from eval import save_predictions`). Đừng tự viết lại định dạng.
5. Trước khi nộp, chạy `eval.py score` cho từng nhóm file và `eval.py grade` cho chung kết so với mốc. Nếu `eval.py` báo lỗi định dạng, sửa code ghi file của bạn, không sửa `eval.py`.

---

## 2. Bước 1: So sánh backbone (≥ 5)

**Mục tiêu:** xem với cùng công thức nền, backbone nào cho macro-F1 val tốt nhất, và đánh đổi giữa chất lượng, kích thước và tốc độ.

### 2.1 Danh sách gợi ý

Chọn **ít nhất 5**, thỏa các ràng buộc sau:

- ít nhất 1 mạng thuộc họ **ResNet** (làm mốc so sánh);
- ít nhất 1 mạng **ResNeXt hoặc ConvNeXt**;
- ít nhất 1 mạng **transformer** (ViT, DeiT hoặc Swin);
- ít nhất 1 mạng **nhẹ** (MobileNetV3, EfficientNet-B0, RegNet nhỏ...).

| Họ | Gợi ý (tên trong `timm`) | Ghi chú từ slide (trang 41–42) |
|---|---|---|
| ResNet | `resnet50` | 25,6M tham số, 4,1 GMAC; mốc mặc định |
| ResNeXt | `resnext50_32x4d` | 25,0M, 4,2 GMAC; thêm cardinality |
| ConvNeXt | `convnext_tiny` | 28,6M, 4,5 GMAC; "ResNet hiện đại hoá" |
| ViT / DeiT | `vit_small_patch16_224` hoặc `deit_small_patch16_224` | DeiT-S 22,1M, 4,6 GMAC; thiên kiến quy nạp yếu hơn CNN |
| Swin | `swin_tiny_patch4_window7_224` | 28,3M, 4,5 GMAC; attention cửa sổ |
| Nhẹ | `efficientnet_b0`, `mobilenetv3_large_100` | EfficientNet-B0 0,39 GMAC, MobileNetV3-L 0,22 GMAC |
| Bonus | DINOv2 ViT-S/B đóng băng + linear probe | Slide (trang 44) gợi ý khi nhãn ít |

**Lưu ý về `timm`:** mỗi kiến trúc có nhiều bộ trọng số (tag) khác nhau, tạo ra từ các công thức huấn luyện khác nhau. Tên và tag có thể đổi theo phiên bản, hãy dùng `timm.list_pretrained('resnet50*')` để xem và **ghi rõ tag đã dùng** trong `results.xlsx`. Slide nhắc: so sánh top-1 mà không ghi công thức là lỗi hay gặp (trang 45). Bạn cũng có thể dùng `torchvision.models`.

### 2.2 Việc phải làm

- Huấn luyện từng backbone bằng **công thức nền ở 1.4**, 1 seed (cùng seed cho mọi backbone).
- Với mỗi backbone ghi: số tham số, GMAC (đếm bằng thư viện hoặc tự đếm), macro-F1 và top-1 trên val, thời gian train mỗi epoch, độ trễ suy luận sơ bộ (batch 1, một lần đo là đủ ở bước này; đo kỹ ở Bước 3).
- Lưu ảnh biểu đồ training của **mỗi** backbone: `curves/B0x_<ten>.png`.
- Chọn **1–2 backbone** để đi tiếp sang Bước 2 và 3. Lý do chọn phải có số liệu (ví dụ: "tốt nhất về macro-F1", hoặc "cân bằng nhất giữa F1 và độ trễ"). Slide (trang 33) nhắc câu hỏi thực tế không phải "mạng nào đứng đầu ImageNet" mà là "mạng nào đủ tốt trong độ trễ đo được".

### 2.3 Gợi ý phân tích

- Backbone nào hội tụ nhanh nhất? Có backbone nào quá khớp rõ rệt (train tăng, val giảm)?
- Thứ hạng backbone trên DeepWeeds có giống thứ hạng trên ImageNet (slide trang 41–42) không? Vì sao có/không?
- FLOPs có dự đoán được thời gian train hoặc độ trễ không? (Slide trang 43: *FLOPs không phải độ trễ*.)

---

## 3. Bước 2: Công thức huấn luyện

**Mục tiêu:** đo từng yếu tố của công thức huấn luyện đóng góp bao nhiêu, trên **1–2 backbone** đã chọn ở Bước 1. Yêu cầu: **ít nhất 3 trục** trong bảng dưới, mỗi trục ít nhất 2 giá trị khác nhau (một trong hai là giá trị của công thức nền).

| Trục | Các giá trị nên thử | Liên hệ slide | Câu hỏi cần trả lời |
|---|---|---|---|
| **A. Khởi tạo** | từ đầu · đóng băng backbone, chỉ train head · tinh chỉnh toàn bộ | Trang 51, 53; Lab #2 | Với ~10k ảnh khác ImageNet, từ đầu có kịp không? Đóng băng có đủ không? |
| **B. Augmentation** | cơ bản (crop + flip) · + đổi màu · TrivialAugment hoặc RandAugment · Mixup · CutMix | Trang 47–50 | Phép nào giúp, phép nào hại? Lật dọc có hợp lệ với ảnh cỏ dại không? |
| **C. Hàm loss** | CE · CE + label smoothing (ε ≈ 0,1) · focal loss (γ ≈ 2) · CE có trọng số theo lớp (hoặc class-balanced) | Trang 56–57 | Với `Negative` áp đảo, loss nào cải thiện F1 các lớp hiếm? |
| **D. Cân bằng mẫu** | không · sampler cân bằng lớp (oversample) | Trang 57 | Sampler khác loss có trọng số như thế nào? |
| **E. LR và optimizer** | LR backbone/head (cùng LR vs gấp 10 lần) · AdamW vs SGD+momentum · có/không warmup · LR theo tầng | Trang 51–52, 55 | Mạng tiền huấn luyện nhạy với LR đến mức nào? |
| **F. Chính quy hoá** | có/không EMA trọng số · weight decay · stochastic depth · dropout head | Trang 56 | EMA có giúp "miễn phí" không? |
| **G. Độ phân giải và thời gian huấn luyện** | 224 vs 256 · 10 vs 20 epoch | Trang 46, 68 | Công thức dài hơn có đáng không? |

### 3.1 Cách làm

1. Gọi cấu hình nền là `T00`. Mỗi lần chạy mới chỉ khác `T00` **một yếu tố** (nguyên tắc N1).
2. Làm theo thứ tự hợp lý: trục có tác dụng lớn (khởi tạo, loss, augmentation) trước, rồi mới tinh chỉnh LR/EMA.
3. Khi một yếu tố **thắng rõ rệt**, bạn được phép đưa nó vào nền cho các thí nghiệm sau (ghi lại nền mới và đánh dấu trong bảng). Cách này gọi là *tham lam theo từng trục*. Nêu rõ trong báo cáo là bạn đã dùng cách này, vì thứ tự các trục có thể ảnh hưởng kết quả.
4. Cố gắng thử **ít nhất một kết hợp** các yếu tố tốt nhất (ví dụ CutMix + label smoothing + EMA) và kiểm tra xem hiệu ứng có **cộng dồn** hay triệt tiêu nhau. Slide (trang 46) cho thấy các bước công thức cộng lại thành khoảng cách lớn, nhưng từng bước chỉ tăng một phần nhỏ.
5. Mỗi thí nghiệm một ảnh biểu đồ: `curves/T0x_<mota>.png`.

### 3.2 Lưu ý kỹ thuật

- **Mixup và CutMix trộn cả nhãn:** loss phải dùng nhãn mềm hoặc tổng có trọng số của hai CE; accuracy lúc train khi dùng Mixup/CutMix không còn nghĩa như bình thường, hãy dựa vào val.
- **Focal loss:** `γ = 0` phải cho lại đúng CE. Hãy tự kiểm tra điều này như một unit test nhỏ.
- **Từ đầu với ViT/Swin:** thiên kiến quy nạp yếu, ít dữ liệu nên thường kém. Đó là kết quả hợp lệ để phân tích, không phải lỗi.
- **Đóng băng:** vẫn phải để BatchNorm ở chế độ eval nếu backbone đóng băng. Hãy nghĩ vì sao.
- **EMA:** đánh giá bằng trọng số EMA, và nếu mạng có BatchNorm thì cân nhắc cập nhật thống kê BN (slide trang 56).
- Giữ **nguyên số epoch** giữa các cấu hình trừ khi đang thử chính yếu tố số epoch.

---

## 4. Bước 3: Phương pháp suy luận

**Mục tiêu:** với mô hình đã huấn luyện xong (không huấn luyện lại), so sánh các cách suy luận về **độ chính xác, hiệu chuẩn và độ trễ**. Yêu cầu **ít nhất 4 phương pháp** khác nhau (không tính 1-view mốc).

**Mẹo quan trọng:** khi chạy ở Bước 1–2, hãy **lưu logit** (tập val và test) ra file `.npy`. Khi đó phần lớn thí nghiệm suy luận chỉ là tính toán trên các file này, không cần GPU. Riêng TTA, đổi độ phân giải và đo độ trễ cần chạy lại model.

| Mã | Phương pháp | Ghi chú |
|---|---|---|
| I00 | **1 view** (mốc) | Resize + center crop, `model.eval()` |
| I01 | **TTA lật ngang** (K = 2) | Slide trang 62, 75 |
| I02 | **TTA nhiều crop hoặc nhiều tỉ lệ** (ví dụ 5 crop, hoặc 3 tỉ lệ) | K càng lớn, chi phí gần tuyến tính theo K (slide trang 63) |
| I03 | **Gộp xác suất vs gộp logit** | Slide chưa có kết luận cái nào luôn tốt hơn: chọn một, ghi rõ, hoặc so sánh cả hai |
| I04 | **Dò độ phân giải kiểm tra** (ví dụ 224 / 256 / 288 / 320) | Slide trang 68 (FixRes): test ở độ phân giải cao hơn lúc train có thể tốt hơn; không đổi tham số, chỉ tăng FLOPs |
| I05 | **Ensemble** vài mô hình (khác backbone hoặc khác seed) | Trung bình xác suất; chi phí = số mô hình (slide trang 67) |
| I06 | **Trọng số EMA** (nếu đã train có EMA) hoặc **model soup** | Không tốn thêm khi suy luận (slide trang 67) |
| I07 | **Temperature scaling + ECE** | Khớp một T duy nhất trên **val**, đo ECE trước và sau. Accuracy không đổi (slide trang 69) |
| I08 | **Gộp BatchNorm vào conv** hoặc **FP16/AMP** | Hiệu chỉnh/độ chính xác có thay đổi không? Đo độ trễ (slide trang 71) |

### 4.1 Đo độ trễ đúng cách

Đây là phần hay làm sai nhất (slide trang 73, 76). Quy tắc:

- **Warmup:** chạy 10 lần đầu bỏ đi (lần đầu tải thư viện, cuBLAS...).
- **Đồng bộ GPU:** GPU chạy bất đồng bộ. Trước và sau đoạn cần đo phải gọi `torch.cuda.synchronize()` (hoặc dùng CUDA event). Dùng `time.time()` mà không đồng bộ sẽ cho số sai.
- **Nhiều lần đo:** ≥ 50 lần, báo cáo **p50, p95, p99**, không chỉ trung bình.
- **Hai điều kiện:** batch 1 (giống robot) và một batch lớn hơn (ví dụ 32, đo thông lượng ảnh/giây).
- **Ghi rõ điều kiện:** tên GPU, độ phân giải, dtype (FP32/AMP/FP16), có/không gộp BN, phiên bản torch.
- **Tính cả tiền xử lý hay không?** Chọn một cách và ghi rõ. Với TTA, thời gian gần bằng K lần một lượt chạy.
- Nhớ rằng ở batch 1 trên một số GPU, AMP có thể **chậm hơn** FP32 (slide trang 73). Hãy kiểm tra trên máy của bạn, đừng giả định.

### 4.2 Tổng hợp

Với mỗi phương pháp ghi: macro-F1 val, top-1 val, ECE (nếu đo), độ trễ p50/p95/p99 ở batch 1, và **chi phí tương đối** so với I00. Từ đó vẽ **đường đánh đổi** độ chính xác và độ trễ (một biểu đồ scatter là đủ). Slide kết luận TTA và ensemble hợp *ngoại tuyến*, còn trên robot nên dùng thứ không tốn thêm (EMA, soup, gộp BN, FP16, độ phân giải đã dò). Dữ liệu của bạn có ủng hộ kết luận này không?

---

## 5. Bước 4: Cấu hình tốt nhất và chạy test

1. **Chốt cấu hình** dựa hoàn toàn trên **val**: backbone, công thức huấn luyện (kết hợp tốt nhất từ Bước 2), phương pháp suy luận (Bước 3).
2. Huấn luyện lại cấu hình đó với **≥ 3 seed khác nhau** (mã `F01`, `F02`, `F03`...). Báo cáo macro-F1 và top-1 val ở dạng mean ± std.
3. Chạy **test đúng một lần cho mỗi seed** trên toàn bộ tập test, cho các cấu hình cuối cùng **và** mốc `T00`/`I00` (cùng số seed, để tính mức cải thiện). Báo cáo mean ± std của macro-F1, top-1, F1 từng lớp (đặc biệt Chinee apple và Snake weed), ECE. **Lưu file dự đoán** `predictions/<exp_id>_seed<k>_test.csv` theo định dạng ở [`README.md` mục 2.2](README.md#22-cách-đánh-giá-bắt-buộc); giảng viên sẽ tính lại chỉ số từ các file này. Chạy `python eval.py score ...` và `python eval.py grade ...` ([`README.md` mục 2.4](README.md#24-công-cụ-evalpy-và-bộ-khung-starter)) để lấy số liệu đưa vào xlsx và báo cáo. Đối chiếu kết quả với số tham khảo ở [`README.md` mục 2.3](README.md#23-số-tham-khảo-từ-bài-báo-gốc), nhớ rằng điều kiện huấn luyện của bài báo (100 epoch, augmentation mạnh) khác bài lab.
4. So sánh với mốc: **cấu hình tốt nhất cải thiện bao nhiêu so với công thức nền + 1-view?** Chênh lệch có lớn hơn std không?
5. Vẽ **ma trận nhầm lẫn** trên test, nêu lớp nào còn nhầm nhiều nhất và đưa ra giả thuyết nguyên nhân (xem vài ảnh bị dự đoán sai).
6. Nếu giảng viên yêu cầu thêm tiêu chí triển khai (ví dụ độ trễ ≤ X ms), hãy nêu **hai** cấu hình: tốt nhất về độ chính xác (ngoại tuyến) và tốt nhất khi có ràng buộc thời gian thực.

> Sau khi mở kết quả test, **không** được quay lại sửa cấu hình rồi chạy test lần nữa. Nếu bạn phát hiện lỗi sau đó, hãy ghi lỗi vào báo cáo và nêu rõ con số test nào bị ảnh hưởng.

---

## 6. Bước 5: Sản phẩm

### 6.1 `results.xlsx`

Một file Excel gồm các sheet sau. Mỗi dòng là một lần chạy hoặc một cấu hình suy luận.

**Quy ước `exp_id`:** `B01…` backbone · `T01…` công thức huấn luyện · `I01…` suy luận · `F01…` chung kết (nhiều seed). `exp_id` phải trùng tên file ảnh trong `curves/` (ví dụ `B01_resnet50.png`).

| Sheet | Cột bắt buộc |
|---|---|
| `Backbones` | exp_id · backbone · tag trọng số · #tham số (M) · GMAC · độ phân giải · epoch · seed · macro-F1 val · top-1 val · thời gian train/epoch · độ trễ batch-1 (ms) · ghi chú |
| `Training` | exp_id · backbone · trục thay đổi (A–G) · khác `T00` ở điểm nào · seed · macro-F1 val · top-1 val · Δ so với `T00` · F1 các lớp hiếm (nếu có) · ghi chú |
| `Inference` | exp_id · phương pháp · mô hình/checkpoint dùng · K (số view hoặc số mô hình) · macro-F1 val · top-1 val · ECE val · độ trễ p50/p95/p99 (ms) batch-1 · thông lượng (ảnh/s) · chi phí tương đối so với `I00` |
| `Final` | exp_id · cấu hình (backbone + công thức + suy luận) · seed · macro-F1 val · macro-F1 test · top-1 test · ECE test · mean ± std qua seed (dòng tổng hợp) |
| `PerClass` | lớp · số ảnh test · precision · recall · F1 (cho cấu hình cuối cùng, dòng mốc và dòng tốt nhất) |
| `Latency` | cấu hình · GPU · dtype · batch · gộp BN (có/không) · p50 · p95 · p99 · ảnh/s |
| `Summary` | **Bảng tổng hợp một trang**: top 10 cấu hình theo macro-F1 val, cùng chi phí/độ trễ. Đây là bảng so sánh chính để đưa vào báo cáo |

Gợi ý trình bày: freeze hàng tiêu đề, định dạng số thập phân thống nhất (ví dụ 4 chữ số), tô nổi bật dòng tốt nhất của mỗi sheet, ghi **đơn vị** trong tên cột. Con số mean ± std có thể để trong hai cột riêng hoặc trong một ô dạng `0.9412 ± 0.0031`, miễn nhất quán.

### 6.2 Ảnh biểu đồ training

Với **mỗi** thí nghiệm huấn luyện (B, T, F) lưu một ảnh `curves/<exp_id>_<mota>.png` gồm tối thiểu:

- loss train và loss val theo epoch,
- macro-F1 (hoặc accuracy) val theo epoch (có thể thêm train),
- (khuyến khích) LR theo bước, để thấy warmup và cosine.

Biểu đồ phải đọc được: có tiêu đề (`exp_id`, backbone), nhãn trục, chú thích. Được dùng matplotlib, TensorBoard hoặc Weights & Biases rồi chụp màn hình, miễn là mỗi thí nghiệm có ảnh riêng và nhìn rõ số liệu.

### 6.3 Dàn ý báo cáo (`report.md` hoặc `report.pdf`, khoảng 4–8 trang)

1. **Tóm tắt (≤ 10 dòng):** bài toán, các thí nghiệm đã làm, cấu hình tốt nhất và con số test cuối cùng (kèm std), kết luận chính.
2. **Dữ liệu và thiết lập:** dataset, fold, phân bố lớp (biểu đồ EDA), chỉ số, công thức nền, phần cứng, phiên bản thư viện, seed.
3. **Kết quả so sánh backbone:** bảng (trích từ `Backbones`), biểu đồ F1 theo độ trễ hoặc tham số, nhận xét.
4. **Kết quả công thức huấn luyện:** bảng ablation theo từng trục, kèm Δ và so sánh với std. Nêu rõ yếu tố nào giúp, yếu tố nào không, và giải thích vì sao (liên hệ slide).
5. **Kết quả suy luận:** bảng, đường đánh đổi độ chính xác–độ trễ, kết quả hiệu chuẩn (ECE trước/sau).
6. **Cấu hình tốt nhất:** mô tả đầy đủ để người khác tái lập, bảng chung kết (mean ± std, test), ma trận nhầm lẫn, phân tích lỗi bằng ảnh.
7. **Kết luận và khuyến nghị:** trả lời trực tiếp các câu hỏi:
   - Cấu hình nào tốt nhất? Tốt hơn mốc bao nhiêu, có vượt nhiễu không?
   - Yếu tố nào đóng góp nhiều nhất: backbone, công thức huấn luyện hay suy luận?
   - Nếu triển khai trên robot với ngân sách 30–100 ms/khung, bạn chọn gì?
8. **Hạn chế và việc tiếp theo:** nêu số seed, một fold, thí nghiệm chưa kiểm chứng, rủi ro lệch phân phối (miền khác, mùa khác...).
9. **Phụ lục:** danh sách `exp_id` và cấu hình đầy đủ, link notebook.

Mọi khẳng định phải dựa vào số trong `results.xlsx`. Dùng ngôn ngữ thận trọng: chênh lệch nhỏ hơn nhiễu thì nói là "không phân biệt được".

---

## 7. Ngân sách tính toán và mẹo Colab/Kaggle

Số lượng lần chạy tối thiểu xấp xỉ: Bước 1 khoảng 5–6 lần, Bước 2 khoảng 10–15 lần, Bước 4 khoảng 3 lần (nhiều seed), chưa kể thử nghiệm hỏng. Tổng khoảng **20–25 lần huấn luyện**. Thời gian mỗi lần phụ thuộc rất nhiều vào backbone (ViT/Swin chậm hơn MobileNet nhiều), độ phân giải và GPU.

**Cách tự ước lượng (đừng tin con số của ai, kể cả tài liệu này):**

1. Chạy 1 epoch của mỗi backbone, ghi thời gian.
2. Nhân với số epoch và số lần chạy dự kiến, rồi so với hạn mức GPU của bạn.
3. Nếu vượt ngân sách, giảm theo thứ tự ưu tiên: (a) epoch 15 → 10; (b) chạy các ablation ở Bước 2 chỉ trên **1 backbone**; (c) chỉ ablation 1 seed, và dành 3 seed cho Bước 4; (d) hạ độ phân giải huấn luyện. **Ghi lại mọi giảm bớt trong báo cáo.**

Mẹo:

- Dùng **AMP**, `pin_memory=True`, `num_workers` hợp lý, và `channels_last` nếu GPU hỗ trợ.
- Ảnh chỉ 256×256, 17,5k ảnh khoảng vài GB nếu giải nén vào RAM: có thể nạp trước để tránh nghẽn đọc đĩa. Hoặc giải nén vào đĩa cục bộ của notebook, không đọc từng ảnh từ Drive.
- **Lưu sau mỗi epoch** (checkpoint và log) để tiếp tục được khi phiên bị ngắt.
- Viết **một hàm train** nhận cấu hình (dict hoặc YAML) rồi chạy hàng loạt, thay vì copy notebook cho từng thí nghiệm.
- Trên Kaggle, dùng *Save & Run All (Commit)* cho các lần chạy dài. Trên Colab, mở tab giữ kết nối và lưu ra Drive.
- Ghi kết quả thẳng vào Drive hoặc `/kaggle/working`, và chỉ commit file nhỏ (xlsx, png, code) vào git.

---

## 8. Bẫy thường gặp

Phần lớn lấy từ chính slide Day 2:

| Bẫy | Hệ quả | Cách tránh |
|---|---|---|
| Quên `model.eval()` khi suy luận | BatchNorm dùng thống kê batch, kết quả phụ thuộc ảnh khác trong batch | Luôn gọi `eval()` và `torch.inference_mode()`/`no_grad()` trước mọi đánh giá (slide trang 14, 61) |
| Chọn mô hình hoặc siêu tham số bằng **test** | Con số test lạc quan, không đại diện | Chỉ val để chọn; test một lần (mục 5) |
| So sánh số của hai lần chạy khác công thức | Kết luận sai về kiến trúc | Cùng công thức, cùng split; ghi rõ khác biệt (slide trang 45) |
| Kết luận từ chênh lệch nhỏ hơn nhiễu | "Phát hiện" không lặp lại được | ≥ 3 seed cho vòng cuối, so sánh với std (slide trang 59) |
| Chỉ nhìn accuracy trên dữ liệu mất cân bằng | Mô hình luôn đoán `Negative` vẫn điểm cao | Dùng macro-F1 và F1 từng lớp |
| Dùng `time.time()` trên GPU không đồng bộ | Độ trễ sai | `torch.cuda.synchronize()` hoặc CUDA event, có warmup (slide trang 73, 76) |
| Coi FP16/INT8 luôn nhanh hơn và không mất chính xác | Chọn sai cấu hình triển khai | Đo thực tế trên máy của bạn (slide trang 73, 76) |
| Coi TTA là "độ chính xác miễn phí" | Bỏ qua chi phí K lần và việc TTA có thể đổi đúng thành sai | Báo cả độ trễ; xem vài ca bị TTA đổi nhãn (slide trang 66) |
| Rò rỉ dữ liệu giữa train, val, test (ví dụ cùng ảnh, hoặc augment trước khi chia) | Điểm cao giả | Chia theo file CSV, không tự chia lại |
| Mixup/CutMix mà vẫn đo accuracy train như bình thường | Hiểu sai đường cong training | Dùng val để đánh giá |
| Đóng băng backbone mà BN vẫn ở train mode | Thống kê BN bị cập nhật, kết quả khó hiểu | Đặt phần đóng băng ở eval mode |
| Đổi nhiều thứ cùng lúc | Không biết yếu tố nào có tác dụng | Nguyên tắc N1 |
| Quên ghi tag trọng số `timm`, version thư viện, seed | Không tái lập được | Ghi vào cấu hình của mỗi lần chạy |

---

## 9. Câu hỏi gợi ý cho phần phân tích

Không bắt buộc trả lời hết, nhưng các câu này giúp báo cáo sâu hơn:

1. Giữa ResNet-50 và ConvNeXt-T (kiến trúc hiện đại hoá), mức chênh lệch thật sự đến từ kiến trúc hay công thức huấn luyện của trọng số tiền huấn luyện? (Slide trang 37.)
2. Với ~10k ảnh, vì sao ViT từ đầu thường kém CNN từ đầu? Dữ liệu của bạn có xác nhận không? (Slide trang 32, 53.)
3. Loss nào cải thiện F1 của lớp hiếm nhất? Cái giá phải trả ở lớp `Negative` là gì?
4. Mixup/CutMix có hợp với ảnh cỏ dại không? Khi nhãn là "loài cỏ" mà ảnh chứa vật thể nhỏ, cắt dán có làm mất vật thể không?
5. Hiệu ứng nào **cộng dồn** khi kết hợp, hiệu ứng nào triệt tiêu?
6. TTA tăng bao nhiêu điểm và tốn bao nhiêu lần độ trễ? Đáng giá hay không, với ràng buộc nào?
7. Tăng độ phân giải kiểm tra có giúp không? Có liên hệ với việc `RandomResizedCrop` khi train không? (FixRes.)
8. Temperature scaling giảm ECE bao nhiêu? Nếu ảnh lúc triển khai khác miền (mùa khác, ánh sáng khác), T khớp trên val còn đáng tin không?
9. Cấu hình nào bạn **sẽ triển khai trên robot** và vì sao?
10. Bạn sẽ làm gì tiếp theo nếu có thêm một ngày: nhiều fold, nhiều dữ liệu, chưng cất (slide trang 58), hay thích ứng lúc kiểm tra?
