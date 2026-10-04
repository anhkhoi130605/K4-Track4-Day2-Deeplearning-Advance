# RUBRIC — Tiêu chí chấm Lab Day 2

**Tổng điểm: 100** (cộng tối đa 10 điểm thưởng, tổng không vượt quá 110).
Điểm thưởng không bù được điều kiện tiên quyết bị vi phạm.

Bài lab chấm **quy trình thực nghiệm** (80 điểm, phần A–H) và **chất lượng model đạt được** (20 điểm, phần I). Một bài có điểm test thấp nhưng thiết kế chặt chẽ, phân tích trung thực vẫn được nhiều điểm ở A–H và G; một bài có điểm test cao nhưng chọn cấu hình bằng test hoặc không có bằng chứng sẽ bị trừ nặng (mục 3) và có thể mất điểm phần I.

---

## 0. Điều kiện tiên quyết

Thiếu một trong các mục sau thì bài **chưa được chấm** (hoặc bị giới hạn điểm, ghi trong ngoặc):

| # | Điều kiện | Hậu quả nếu thiếu |
|---|---|---|
| P1 | Có đủ 6 sản phẩm theo `README.md` mục 4: `results.xlsx`, báo cáo, `curves/`, `code/`, README riêng với link chạy lại, `predictions/` | Trả bài, yêu cầu bổ sung |
| P2 | Số liệu đến từ lần chạy thật; truy ngược được `exp_id` → log/ảnh biểu đồ | Phần có số liệu bịa hoặc không truy ngược được **bị 0 điểm** và báo cáo vi phạm học thuật |
| P3 | Dùng đúng dataset DeepWeeds và **fold 0 chia sẵn** theo quy tắc S1–S6 (`README.md` mục 2.1) | Giới hạn tối đa 60 điểm nếu tự chia lại, gộp val vào train hoặc sửa file CSV mà không được giảng viên đồng ý |
| P4 | `predictions/` đủ cho cấu hình chung kết và mốc, mọi seed, đúng định dạng `README.md` mục 2.2 | **Phần I bị 0 điểm** (giảng viên không tính lại được chỉ số) |

---

## 1. Bảng điểm

### A. Thiết lập và tính chặt chẽ của pipeline (12 điểm)

| Tiêu chí | Điểm |
|---|---|
| Chia dữ liệu đúng (fold 0), có chạy và báo cáo các kiểm tra ở `README.md` mục 2.1 (số ảnh từng tập/lớp, giao rỗng, hợp đủ 17.509 ảnh); chọn mô hình/siêu tham số **chỉ trên val**; test chạy **một lần mỗi seed** ở cuối, có nêu rõ trong báo cáo | 4 |
| EDA: phân bố lớp (biểu đồ), ảnh mẫu, đối chiếu với Table 1 của bài báo, nhận xét về mất cân bằng | 2 |
| Kiểm tra pipeline trước khi chạy thật: loss ban đầu ≈ ln 9, overfit 1 batch nhỏ, kiểm tra ảnh sau augmentation (có bằng chứng trong notebook hoặc báo cáo) | 3 |
| Cố định seed, ghi cấu hình đầy đủ (tag trọng số, hyperparameter, version thư viện); dùng `eval()` đúng lúc | 3 |

### B. So sánh backbone (12 điểm)

| Tiêu chí | Điểm |
|---|---|
| **≥ 5 backbone**, đủ ràng buộc: có ResNet, có ResNeXt hoặc ConvNeXt, có ít nhất 1 transformer, có ít nhất 1 mạng nhẹ (thiếu mỗi ràng buộc trừ 1 điểm; dưới 5 backbone: tối đa 5/12 cho mục này) | 4 |
| **Công bằng:** cùng công thức nền, cùng split, cùng seed; ghi rõ tag trọng số | 3 |
| Báo cáo đủ: params, GMAC, macro-F1/top-1 val, thời gian train, độ trễ sơ bộ | 3 |
| Chọn backbone đi tiếp **có lý do dựa trên số liệu** (không chỉ "vì F1 cao nhất" nếu có đánh đổi khác) | 2 |

### C. Công thức huấn luyện (16 điểm)

| Tiêu chí | Điểm |
|---|---|
| **≥ 3 trục** (khởi tạo, augmentation, loss, sampler, LR/optimizer, chính quy hoá, độ phân giải...), mỗi trục ≥ 2 giá trị; **ít nhất một trục là loss hoặc augmentation** | 5 |
| **Thiết kế có kiểm soát:** mỗi lần chạy chỉ khác nền một yếu tố, bảng ghi rõ khác ở điểm nào; nếu dùng cách tham lam theo trục thì nêu rõ | 4 |
| Cài đặt đúng các kỹ thuật (ví dụ focal `γ=0` ≡ CE; Mixup/CutMix trộn cả nhãn; weight decay không áp dụng cho norm/bias; EMA đánh giá đúng trọng số) | 3 |
| Thử **ít nhất một kết hợp** các yếu tố tốt và nhận xét cộng dồn hay triệt tiêu | 1 |
| Phân tích so sánh Δ với nhiễu (std); không kết luận từ chênh lệch nhỏ hơn nhiễu | 3 |

### D. Suy luận (12 điểm)

| Tiêu chí | Điểm |
|---|---|
| **≥ 4 phương pháp suy luận** ngoài mốc 1-view (ví dụ: TTA lật, multi-crop/scale, dò độ phân giải, gộp xác suất vs logit, ensemble, EMA/soup, temperature scaling, gộp BN/FP16) | 4 |
| Đo độ trễ **đúng cách**: warmup, `cuda.synchronize` (hoặc CUDA event), ≥ 50 lần, báo cáo p50/p95/p99, ghi rõ GPU/dtype/batch | 3 |
| Có **hiệu chuẩn:** ECE trước và sau temperature scaling, T khớp trên val | 2 |
| Biểu đồ/bảng **đánh đổi độ chính xác và độ trễ**, nhận xét phương pháp nào hợp ngoại tuyến, phương pháp nào hợp thời gian thực | 3 |

### E. Bảng so sánh `results.xlsx` (8 điểm)

| Tiêu chí | Điểm |
|---|---|
| Đủ các sheet và cột bắt buộc (`GUIDE.md` mục 6.1); mọi thí nghiệm có mặt | 3 |
| Nhất quán: `exp_id` khớp ảnh biểu đồ, file dự đoán và báo cáo; đơn vị rõ; mean ± std đúng cho cấu hình chung kết | 3 |
| Sheet `Summary` rõ ràng, dễ đọc, làm nổi bật cấu hình tốt nhất và so sánh với mốc | 2 |

### F. Biểu đồ training (4 điểm)

| Tiêu chí | Điểm |
|---|---|
| Mỗi thí nghiệm huấn luyện (B, T, F) có một ảnh riêng, đủ loss train/val và metric val theo epoch | 2 |
| Đọc được: tiêu đề, nhãn trục, chú thích, tên khớp `exp_id`; có nhận xét về hiện tượng đáng chú ý (quá khớp, nhiễu, hội tụ chậm...) trong báo cáo | 2 |

### G. Báo cáo kết luận (12 điểm)

| Tiêu chí | Điểm |
|---|---|
| Đủ cấu trúc theo `GUIDE.md` mục 6.3; tóm tắt nêu rõ cấu hình tốt nhất và con số test cuối cùng kèm std | 3 |
| **Kết luận có bằng chứng:** trả lời được yếu tố nào đóng góp nhiều nhất (backbone, huấn luyện hay suy luận) và cấu hình nào tốt nhất, mỗi khẳng định gắn với số trong xlsx | 4 |
| **Phân tích lỗi:** ma trận nhầm lẫn, F1 từng lớp, xem ảnh bị đoán sai (đặc biệt cặp Chinee apple ↔ Snake weed) và đưa ra giả thuyết | 3 |
| **Trung thực và hạn chế:** nêu số seed, một fold, chia ngẫu nhiên (không theo địa điểm) có thể làm điểm test lạc quan, giảm bớt do ngân sách GPU, thí nghiệm thất bại, rủi ro lệch phân phối; khuyến nghị triển khai thực tế | 2 |

### H. Code và khả năng tái lập (4 điểm)

| Tiêu chí | Điểm |
|---|---|
| Hoàn thiện đúng và đầy đủ bộ khung `starter/` (không còn `NotImplementedError`; có kiểm tra tự viết cho các phần dễ sai như focal `γ=0`, CutMix, gộp BN); một hàm train dùng chung cho mọi cấu hình; code đọc được. Không sửa `eval.py` | 2 |
| README riêng có link notebook Colab/Kaggle, version thư viện, thứ tự chạy; người khác chạy lại được và ra kết quả cùng mức | 2 |

### I. Chất lượng model đạt được (20 điểm)

**Cách chấm:** giảng viên **tính lại chỉ số từ `predictions/`** (định nghĩa ở `README.md` mục 2.2), trên **toàn bộ tập test fold 0**, cho **cấu hình chung kết**, **trung bình qua các seed** (≥ 3 seed). Số trong báo cáo hoặc xlsx lệch khỏi số tính lại thì lấy số tính lại, và áp dụng mục 3 nếu lệch không giải thích được.

Các ngưỡng neo vào số công bố trong bài báo gốc (`README.md` mục 2.3). Bài báo huấn luyện khoảng 100 epoch với augmentation mạnh, còn bài lab chỉ 10–15 epoch, nên các bậc dưới mốc 95,1% vẫn có điểm. **Các ngưỡng này là ngưỡng tạm thời; giảng viên có thể điều chỉnh và sẽ thông báo trước hạn nộp.**

| # | Tiêu chí | Cách tính | Điểm |
|---|---|---|---|
| I1 | **Top-1 accuracy trên test** (mean qua seed) | ≥ 95,7%: **7** · ≥ 95,1%: **6** · ≥ 94,0%: **5** · ≥ 92,0%: **3** · ≥ 90,0%: **1** · < 90,0%: **0**. (Mốc 95,7% và 95,1% là ResNet-50 và Inception-v3 của bài báo) | 7 |
| I2 | **Macro-F1 test cải thiện so với mốc của chính bạn** (`T00` + `I00`, cùng số seed). Gọi Δ = macro-F1 trung bình của chung kết trừ của mốc, s = std lớn hơn trong hai nhóm seed | Δ ≤ 0: **0** · 0 < Δ ≤ s: **2** · Δ > s: **4** · Δ > s và Δ ≥ 0,01: **5** | 5 |
| I3 | **Hai lớp khó: Chinee apple và Snake weed** (recall trên test, mean qua seed). Mốc bài báo: 88,5% và 88,8% | Cả hai ≥ mốc: **4** · cả hai ≥ 85%: **3** · cả hai ≥ 80%: **2** · có báo cáo đầy đủ nhưng dưới 80%: **1** | 4 |
| I4 | **Ổn định và hiệu chuẩn:** (a) ECE test sau temperature scaling (T khớp trên val) nhỏ hơn ECE trước; (b) chênh lệch macro-F1 giữa val và test của chung kết không quá 0,02, hoặc có giải thích | Mỗi ý đạt được **1** | 2 |
| I5 | **Cấu hình dùng được cho thời gian thực:** nêu một cấu hình có độ trễ **p95 ≤ 100 ms ở batch 1** trên phần cứng bạn khai báo (ngân sách một chu kỳ cảm biến, slide trang 61), kèm macro-F1 test của cấu hình đó | Có và đo đúng cách: **2** · có nhưng đo thiếu (không warmup/synchronize): **1** | 2 |

Ghi chú:

- Bài thất bại ở I1–I3 vẫn được điểm cho phân tích ở phần G nếu trung thực và có bằng chứng.
- Không chấm I1–I3 theo kết quả của một seed đơn lẻ; dùng mean qua seed. Báo cáo kèm std.
- Số theo lớp của bài báo được coi là tương đương recall theo lớp. Bài báo ghi "weighted average accuracy"; I1 dùng top-1 accuracy không trọng số, nên so sánh chỉ mang tính tham chiếu.
- Điểm I chỉ tính trên test. Một lần chạy test lặp lại để chọn kết quả tốt hơn bị xử lý theo mục 3, không chỉ mất điểm I.

---

## 2. Điểm thưởng (tối đa +10)

| Việc làm thêm | Điểm |
|---|---|
| Linear probe với DINOv2 (hoặc mô hình nền tảng khác) đóng băng, so sánh với CNN tinh chỉnh | +2 |
| Chạy nhiều fold (ví dụ 3–5 fold, mỗi fold dùng đủ bộ ba file của fold đó) cho cấu hình cuối, báo cáo mean ± std qua fold | +3 |
| Chưng cất tri thức (giáo viên lớn → học sinh nhỏ) và so sánh với huấn luyện thường | +2 |
| Grad-CAM hoặc bản đồ attention để giải thích lỗi | +1 |
| Test-time adaptation đơn giản (chuẩn hoá lại thống kê BN, hoặc Tent) trên tập lệch miền tự tạo (ví dụ ảnh làm tối/nhiễu), **không** dùng test fold 0 để thích ứng | +2 |
| Xuất ONNX và so sánh độ trễ với PyTorch | +1 |
| Phân tích riêng về lệch phân phối: đánh giá trên ảnh bị nhiễu/làm mờ/thiếu sáng, ECE trước/sau | +2 |

Điểm thưởng chỉ tính khi phần làm thêm có số liệu thật và nằm trong báo cáo.

---

## 3. Lỗi bị trừ điểm

| Lỗi | Trừ |
|---|---|
| Chọn cấu hình, siêu tham số hoặc phương pháp suy luận **dựa vào điểm test**, hoặc chạy test nhiều lần rồi chọn lần tốt hơn | −10 đến −20 (nghiêm trọng, tuỳ mức) |
| Gộp val vào train, huấn luyện trên test, hoặc dùng thông tin từ test (thống kê, ngưỡng, nhiệt độ T) | −10 đến −20 (nghiêm trọng, tuỳ mức), đồng thời áp dụng P3 |
| Thiếu `model.eval()` khi đánh giá, hoặc đo độ trễ không `synchronize` / không warmup (số liệu sai) | −3 mỗi loại, tối đa −6 |
| Kết luận "A tốt hơn B" khi chênh lệch nhỏ hơn std hoặc chỉ có 1 seed mà không nêu hạn chế | −1 mỗi chỗ, tối đa −5 |
| Chỉ báo accuracy, không có macro-F1 hoặc F1 từng lớp | −3 |
| Thí nghiệm không có ảnh biểu đồ training | −0,5 mỗi thí nghiệm, tối đa −5 |
| Số liệu trong báo cáo/xlsx mâu thuẫn với số tính lại từ `predictions/` hoặc với nhau | −1 mỗi chỗ, tối đa −5 |
| Commit dataset hoặc checkpoint lớn vào git | −2 |
| Nộp trễ | Theo quy định của giảng viên |
| **Bịa số liệu, chép bài, hoặc chép số từ bài báo/slide như kết quả của mình** | **0 điểm phần liên quan; báo cáo vi phạm học thuật** |

---

## 4. Mức điểm tham khảo

| Điểm | Mô tả |
|---|---|
| **90–100+** | Đủ ≥ 5 backbone, ≥ 3 trục huấn luyện có kiểm soát, ≥ 4 phương pháp suy luận với độ trễ đo đúng; cấu hình cuối ≥ 3 seed; test một lần mỗi seed; kết quả test đạt các bậc cao ở phần I; báo cáo kết luận có bằng chứng và phân tích lỗi; code tái lập được |
| **75–89** | Đạt các ngưỡng tối thiểu, nhưng thiếu một số điểm chặt chẽ (ví dụ ít seed ở vòng cuối, phân tích nông, bảng thiếu cột) hoặc kết quả test ở bậc trung bình |
| **60–74** | Đủ sản phẩm nhưng một phần thí nghiệm thiếu kiểm soát, kết luận không đủ bằng chứng, hoặc kết quả test thấp |
| **< 60** | Thiếu ngưỡng tối thiểu (số backbone, số trục, số phương pháp), hoặc vi phạm quy tắc val/test, hoặc không tái lập được |

---

## 5. Danh sách tự kiểm trước khi nộp

- [ ] Dùng đúng fold 0, không sửa CSV, không gộp val vào train; đã chạy kiểm tra giao rỗng và hợp đủ 17.509 ảnh.
- [ ] ≥ 5 backbone, cùng công thức nền, ghi tag trọng số.
- [ ] ≥ 3 trục công thức huấn luyện, mỗi lần chạy khác nền một yếu tố.
- [ ] ≥ 4 phương pháp suy luận, độ trễ p50/p95/p99 đo đúng cách.
- [ ] Cấu hình cuối và mốc (`T00` + `I00`) chạy ≥ 3 seed, báo cáo mean ± std; test chạy **một lần mỗi seed**.
- [ ] `predictions/<exp_id>_seed<k>_test.csv` đủ cho chung kết và mốc, mọi seed, đúng cột.
- [ ] `python eval.py score` chạy không lỗi trên mọi nhóm file dự đoán; macro-F1, top-1, recall Chinee apple và Snake weed trong báo cáo khớp kết quả của `eval.py`.
- [ ] Đã chạy `python eval.py grade` và xem kết quả phần I (đề xuất).
- [ ] Không còn `NotImplementedError` trong `code/`; không sửa `eval.py`.
- [ ] Đã nêu một cấu hình có p95 ≤ 100 ms ở batch 1 (hoặc giải thích vì sao không có).
- [ ] `results.xlsx` đủ sheet, `exp_id` khớp ảnh trong `curves/`.
- [ ] Mỗi thí nghiệm huấn luyện có ảnh biểu đồ riêng.
- [ ] Báo cáo có tóm tắt, bảng so sánh, ma trận nhầm lẫn, kết luận, hạn chế (nêu chia ngẫu nhiên có thể lạc quan).
- [ ] Code đầy đủ, README riêng có link notebook chạy lại được.
- [ ] Không commit dataset hoặc checkpoint lớn.
- [ ] Mọi số liệu đến từ lần chạy thật của bạn.
