"""Kiểm tra eval.py bằng dữ liệu giả: so với scikit-learn, so với giá trị tính tay.

Chạy từ thư mục gốc repo:
    python -m unittest discover -s tests -v
(scikit-learn chỉ cần để chạy test này, eval.py không cần nó.)
"""
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix as sk_cm,
                             f1_score, precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval as ev  # noqa: E402

K = ev.NUM_CLASSES


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def make_data(n=500, seed=0, strength=2.0):
    """Nhãn thật và xác suất giả; strength càng lớn càng đoán đúng."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), y] += strength
    return y, softmax(logits)


def write_pred(path, y_true, probs, names=None, y_pred=None):
    n = len(y_true)
    names = names if names is not None else [f"img{i:05d}.jpg" for i in range(n)]
    df = pd.DataFrame({"Filename": names, "y_true": y_true,
                       "y_pred": probs.argmax(1) if y_pred is None else y_pred})
    for i in range(K):
        df[f"p{i}"] = probs[:, i]
    df.to_csv(path, index=False)


class TestMetricsAgainstSklearn(unittest.TestCase):
    def test_all_metrics_match_sklearn(self):
        for seed in range(5):
            y, p = make_data(seed=seed)
            yp = p.argmax(1)
            m = ev.compute_metrics(y, yp, p)
            labels = list(range(K))
            self.assertAlmostEqual(m["top1"], accuracy_score(y, yp), places=12)
            self.assertAlmostEqual(m["macro_f1"], f1_score(y, yp, average="macro", labels=labels, zero_division=0), places=12)
            self.assertAlmostEqual(m["balanced_acc"], balanced_accuracy_score(y, yp), places=12)
            pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=labels, zero_division=0)
            np.testing.assert_allclose(m["precision"], pr, atol=1e-12)
            np.testing.assert_allclose(m["recall"], rc, atol=1e-12)
            np.testing.assert_allclose(m["f1"], f1, atol=1e-12)
            np.testing.assert_array_equal(m["support"], sup)
            np.testing.assert_array_equal(m["confusion"], sk_cm(y, yp, labels=labels))

    def test_class_never_predicted_gives_zero_f1(self):
        y = np.array([0, 0, 1, 1, 2, 2] * 5)
        yp = np.array([0, 0, 1, 1, 1, 1] * 5)  # lớp 2 không bao giờ được đoán
        p = np.full((len(y), K), 1e-3)
        p[np.arange(len(y)), yp] = 1.0
        p /= p.sum(1, keepdims=True)
        m = ev.compute_metrics(y, yp, p)
        self.assertEqual(m["f1"][2], 0.0)
        self.assertEqual(m["recall"][2], 0.0)


class TestECE(unittest.TestCase):
    def test_hand_computed_single_bin(self):
        # 10 mẫu, độ tin cậy 0.9, đúng 5/10 -> ECE = |0.5 - 0.9| = 0.4
        p = np.full((10, K), 0.1 / (K - 1))
        p[:, 0] = 0.9
        y = np.array([0] * 5 + [1] * 5)
        self.assertAlmostEqual(ev.ece_score(p, y), 0.4, places=12)

    def test_hand_computed_two_bins(self):
        # 4 mẫu conf 0.9 (đúng 4/4) + 4 mẫu conf 0.5 (đúng 2/4)
        # ECE = 0.5*|1-0.9| + 0.5*|0.5-0.5| = 0.05
        rows, y = [], []
        for correct in (1, 1, 1, 1):
            r = np.full(K, 0.1 / (K - 1)); r[0] = 0.9; rows.append(r); y.append(0 if correct else 1)
        for correct in (1, 1, 0, 0):
            r = np.full(K, 0.5 / (K - 1)); r[0] = 0.5; rows.append(r); y.append(0 if correct else 1)
        self.assertAlmostEqual(ev.ece_score(np.array(rows), np.array(y)), 0.05, places=12)

    def test_perfect_confidence_all_correct_is_zero(self):
        p = np.eye(K)[np.arange(K)]
        self.assertAlmostEqual(ev.ece_score(p, np.arange(K)), 0.0, places=12)

    def test_bin_boundary_is_right_inclusive(self):
        # conf đúng bằng biên 1/15 phải rơi vào bin đầu (0, 1/15], không vào bin thứ hai
        conf = np.array([1 / 15, 1 / 15 + 1e-9])
        idx = np.clip(np.ceil(conf * 15).astype(int) - 1, 0, 14)
        self.assertEqual(list(idx), [0, 1])


class TestAggregation(unittest.TestCase):
    def test_std_uses_ddof_1(self):
        mean, std = ev.mean_std([0.90, 0.92, 0.94])
        self.assertAlmostEqual(mean, 0.92)
        self.assertAlmostEqual(std, 0.02)
        self.assertAlmostEqual(std, float(np.std([0.90, 0.92, 0.94], ddof=1)))

    def test_single_seed_has_nan_std(self):
        mean, std = ev.mean_std([0.9])
        self.assertTrue(math.isnan(std))

    def test_vector_mean_std(self):
        mean, std = ev.mean_std([[1.0, 2.0], [3.0, 6.0]])
        np.testing.assert_allclose(mean, [2.0, 4.0])
        np.testing.assert_allclose(std, [math.sqrt(2), math.sqrt(8)])


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.y, self.p = make_data(n=100)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_file_loads(self):
        f = self.dir / "a_seed0_test.csv"
        write_pred(f, self.y, self.p)
        pred = ev.read_pred(str(f))
        self.assertEqual(pred.seed, 0)
        self.assertEqual(len(pred.y_true), 100)

    def test_rejects_probs_not_summing_to_one(self):
        f = self.dir / "a_seed0_test.csv"
        write_pred(f, self.y, self.p * 0.5)
        with self.assertRaisesRegex(ValueError, "tổng xác suất"):
            ev.read_pred(str(f))

    def test_rejects_raw_logits(self):
        f = self.dir / "a_seed0_test.csv"
        logits = np.log(self.p)  # logit giả: toàn số âm
        write_pred(f, self.y, logits, y_pred=self.p.argmax(1))
        with self.assertRaisesRegex(ValueError, "lưu softmax"):
            ev.read_pred(str(f))

    def test_rejects_pred_not_argmax(self):
        f = self.dir / "a_seed0_test.csv"
        yp = self.p.argmax(1).copy()
        yp[0] = (yp[0] + 1) % K
        write_pred(f, self.y, self.p, y_pred=yp)
        with self.assertRaisesRegex(ValueError, "argmax"):
            ev.read_pred(str(f))

    def test_rejects_missing_column(self):
        f = self.dir / "a_seed0_test.csv"
        write_pred(f, self.y, self.p)
        pd.read_csv(f).drop(columns=["p3"]).to_csv(f, index=False)
        with self.assertRaisesRegex(ValueError, "thiếu cột"):
            ev.read_pred(str(f))

    def test_rejects_duplicate_filename(self):
        f = self.dir / "a_seed0_test.csv"
        names = [f"img{i % 50}.jpg" for i in range(100)]
        write_pred(f, self.y, self.p, names=names)
        with self.assertRaisesRegex(ValueError, "trùng"):
            ev.read_pred(str(f))

    def test_check_against_csv(self):
        f = self.dir / "a_seed0_test.csv"
        names = [f"img{i:05d}.jpg" for i in range(100)]
        write_pred(f, self.y, self.p, names=names)
        ref = self.dir / "test.csv"
        pd.DataFrame({"Filename": names, "Label": self.y, "Species": "x"}).to_csv(ref, index=False)
        ev.check_against_csv(ev.read_pred(str(f)), str(ref))  # khớp: không lỗi

        bad = pd.DataFrame({"Filename": names, "Label": (self.y + 1) % K, "Species": "x"})
        bad_ref = self.dir / "bad.csv"
        bad.to_csv(bad_ref, index=False)
        with self.assertRaisesRegex(ValueError, "y_true khác Label"):
            ev.check_against_csv(ev.read_pred(str(f)), str(bad_ref))

        short = self.dir / "short.csv"
        pd.DataFrame({"Filename": names[:90], "Label": self.y[:90], "Species": "x"}).to_csv(short, index=False)
        with self.assertRaisesRegex(ValueError, "không khớp"):
            ev.check_against_csv(ev.read_pred(str(f)), str(short))

    def test_duplicate_seed_rejected(self):
        for name in ("a_seed0_test.csv", "b_seed0_test.csv"):
            write_pred(self.dir / name, self.y, self.p)
        with self.assertRaisesRegex(ValueError, "seed bị trùng"):
            ev.load_group(str(self.dir / "*_seed0_test.csv"), None)


class TestPoints(unittest.TestCase):
    def test_i1_boundaries(self):
        cases = {95.7: 7, 95.69: 6, 95.1: 6, 95.09: 5, 94.0: 5, 93.99: 3, 92.0: 3,
                 91.99: 1, 90.0: 1, 89.99: 0}
        for acc, pts in cases.items():
            self.assertEqual(ev.points_i1(acc), pts, acc)

    def test_i2_rules(self):
        self.assertEqual(ev.points_i2(-0.01, 0.002), 0)
        self.assertEqual(ev.points_i2(0.0, 0.002), 0)
        self.assertEqual(ev.points_i2(0.001, 0.002), 2)   # 0 < delta <= s
        self.assertEqual(ev.points_i2(0.005, 0.002), 4)   # delta > s nhưng < 0.01
        self.assertEqual(ev.points_i2(0.012, 0.002), 5)   # delta > s và >= 0.01
        self.assertEqual(ev.points_i2(0.012, 0.02), 2)    # delta <= s -> chỉ 2
        self.assertEqual(ev.points_i2(float("nan"), 0.1), 0)

    def test_i3_tiers(self):
        self.assertEqual(ev.points_i3({"Chinee Apple": 88.5, "Snake Weed": 88.8}), 4)
        self.assertEqual(ev.points_i3({"Chinee Apple": 88.4, "Snake Weed": 95.0}), 3)  # cả hai >= 85
        self.assertEqual(ev.points_i3({"Chinee Apple": 85.0, "Snake Weed": 84.9}), 2)
        self.assertEqual(ev.points_i3({"Chinee Apple": 79.0, "Snake Weed": 99.0}), 1)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.n = 600
        self.names = [f"img{i:05d}.jpg" for i in range(self.n)]
        rng = np.random.default_rng(7)
        self.y = rng.integers(0, K, self.n)
        pd.DataFrame({"Filename": self.names, "Label": self.y, "Species": "x"}).to_csv(
            self.dir / "test.csv", index=False)
        labels = pd.DataFrame({"Filename": ["a"] * K, "Label": range(K), "Species": ev.CLASS_NAMES})
        labels.to_csv(self.dir / "labels.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_group(self, prefix, strength, seeds=(0, 1, 2), temperature=1.0, suffix="test"):
        for s in seeds:
            rng = np.random.default_rng(100 + s)
            logits = rng.normal(size=(self.n, K))
            logits[np.arange(self.n), self.y] += strength
            write_pred(self.dir / f"{prefix}_seed{s}_{suffix}.csv", self.y, softmax(logits / temperature), self.names)

    def run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ev.main(argv)
        return rc, buf.getvalue()

    def test_score_end_to_end(self):
        self._write_group("F01", 3.0)
        rc, out = self.run_cli(["score", "--pred", str(self.dir / "F01_seed*_test.csv"),
                                "--test-csv", str(self.dir / "test.csv"),
                                "--labels", str(self.dir / "labels.csv"),
                                "--tag", "F01", "--out", str(self.dir / "out")])
        self.assertEqual(rc, 0)
        self.assertIn("macro-F1", out)
        self.assertIn("Chinee Apple", out)
        self.assertTrue((self.dir / "out" / "F01_summary.json").exists())
        self.assertTrue((self.dir / "out" / "F01_confusion_sum.csv").exists())

    def test_score_reports_error_with_exit_code_2(self):
        self._write_group("F01", 3.0)
        pd.DataFrame({"Filename": self.names[:10], "Label": self.y[:10], "Species": "x"}).to_csv(
            self.dir / "short.csv", index=False)
        rc, _ = self.run_cli(["score", "--pred", str(self.dir / "F01_seed*_test.csv"),
                              "--test-csv", str(self.dir / "short.csv")])
        self.assertEqual(rc, 2)

    def test_grade_end_to_end(self):
        # Mô hình giả có accuracy ~98% nên T=0.4 gần hiệu chuẩn (ECE ~0.02), T=3 rất thiếu tự tin.
        self._write_group("F01", 4.0, temperature=0.4)           # chung kết (đã hiệu chuẩn)
        self._write_group("F01uncal", 4.0, temperature=3.0)      # chưa hiệu chuẩn: ECE lớn
        self._write_group("T00", 2.0)                              # mốc yếu hơn
        self._write_group("F01", 4.0, temperature=0.4, suffix="val")
        rc, out = self.run_cli([
            "grade", "--final", str(self.dir / "F01_seed*_test.csv"),
            "--baseline", str(self.dir / "T00_seed*_test.csv"),
            "--uncal", str(self.dir / "F01uncal_seed*_test.csv"),
            "--final-val", str(self.dir / "F01_seed*_val.csv"),
            "--latency-p95-ms", "42.0", "--latency-method", "proper",
            "--test-csv", str(self.dir / "test.csv"),
            "--labels", str(self.dir / "labels.csv"), "--out", str(self.dir / "out")])
        self.assertEqual(rc, 0, out)
        self.assertIn("I1", out)
        self.assertIn("Tổng các ý đã chấm", out)
        report = json.loads((self.dir / "out" / "grade_I.json").read_text())
        items = {i["code"]: i for i in report["items"]}
        self.assertEqual(items["I2"]["points"], 5)    # mốc yếu hơn rõ rệt
        self.assertEqual(items["I4a"]["points"], 1)   # ECE sau < trước
        self.assertEqual(items["I5"]["points"], 2)

    def test_grade_i4a_zero_when_calibration_gets_worse(self):
        self._write_group("F01", 4.0, temperature=3.0)           # "sau" TS lại tệ hơn
        self._write_group("F01uncal", 4.0, temperature=0.4)
        self._write_group("T00", 2.0)
        rc, _ = self.run_cli(["grade", "--final", str(self.dir / "F01_seed*_test.csv"),
                              "--baseline", str(self.dir / "T00_seed*_test.csv"),
                              "--uncal", str(self.dir / "F01uncal_seed*_test.csv"),
                              "--out", str(self.dir / "out")])
        self.assertEqual(rc, 0)
        report = json.loads((self.dir / "out" / "grade_I.json").read_text())
        items = {i["code"]: i for i in report["items"]}
        self.assertEqual(items["I4a"]["points"], 0)

    def test_grade_flags_missing_optional_inputs(self):
        self._write_group("F01", 4.0)
        self._write_group("T00", 2.0)
        rc, out = self.run_cli(["grade", "--final", str(self.dir / "F01_seed*_test.csv"),
                                "--baseline", str(self.dir / "T00_seed*_test.csv")])
        self.assertEqual(rc, 0)
        self.assertIn("chưa chấm được", out)
        self.assertIn("Chưa chấm được: I4a, I4b, I5", out)

    def test_grade_warns_on_too_few_seeds(self):
        self._write_group("F01", 4.0, seeds=(0, 1))
        self._write_group("T00", 2.0)
        rc, out = self.run_cli(["grade", "--final", str(self.dir / "F01_seed*_test.csv"),
                                "--baseline", str(self.dir / "T00_seed*_test.csv")])
        self.assertEqual(rc, 0)
        self.assertIn("chỉ có 2 seed", out)


if __name__ == "__main__":
    unittest.main()
