import unittest
import tempfile
import os
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from spiralreality_AIT_onepass_aifcore_integrated.integrated.benchmark import (
    load_benchmark_split, run_benchmark, segmentation_f1, split_corpus,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT
from scripts.run_evaluation import _perturb_text, run_evaluation
import random


class HeldOutSplitTest(unittest.TestCase):
    @staticmethod
    def corpus():
        texts = [f"{lang} sample {index}" for lang in ("en", "es", "ja") for index in range(4)]
        segments = [[text[:3], text[3:]] for text in texts]
        languages = [text[:2] for text in texts]
        return texts, segments, languages

    def test_split_is_stratified_disjoint_and_independent_of_input_order(self):
        texts, segments, languages = self.corpus()
        first = split_corpus(texts, segments, languages, seed=12)
        reordered = split_corpus(texts[::-1], segments[::-1], languages[::-1], seed=12)
        self.assertEqual(first, reordered)
        self.assertFalse(set(first.train_texts) & set(first.test_texts))
        self.assertEqual(set(first.train_texts + first.test_texts), set(texts))
        self.assertEqual(set(first.train_languages), {"en", "es", "ja"})
        self.assertEqual(set(first.test_languages), {"en", "es", "ja"})
        self.assertEqual(first.receipt["test"]["samples"], 3)
        self.assertFalse(set(first.receipt["train_pool"]["text_sha256"]) & set(first.receipt["test"]["text_sha256"]))

    def test_exact_duplicates_are_removed_before_external_and_internal_split(self):
        texts, segments, languages = self.corpus()
        clean = split_corpus(texts, segments, languages)
        repeated = split_corpus(texts + texts[:3], segments + segments[:3], languages + languages[:3])
        self.assertEqual(clean.train_texts, repeated.train_texts)
        self.assertEqual(clean.test_texts, repeated.test_texts)
        self.assertEqual(repeated.receipt["duplicate_samples_removed"], 3)
        self.assertEqual(repeated.receipt["selected_samples"], 12)

    def test_cap_balances_languages_and_reports_omissions(self):
        texts, segments, languages = self.corpus()
        split = split_corpus(texts, segments, languages, max_samples=4)
        self.assertEqual(len(split.train_texts), 2)
        self.assertEqual(len(split.test_texts), 2)
        self.assertEqual(set(split.train_languages), set(split.test_languages))
        self.assertEqual(len(set(split.train_languages)), 2)
        self.assertEqual(len(split.receipt["omitted_languages"]), 1)

    def test_invalid_and_tiny_corpora_fail_explicitly(self):
        texts, segments, languages = self.corpus()
        for fraction in (0, 1, -0.1, float("nan")):
            with self.subTest(test_fraction=fraction), self.assertRaises(ValueError):
                split_corpus(texts, segments, languages, test_fraction=fraction)
        for cap in (0, 1, -1, 1.5, True):
            with self.subTest(max_samples=cap), self.assertRaises(ValueError):
                split_corpus(texts, segments, languages, max_samples=cap)
        for args in (([], [], []), (["ab"], [["a", "b"]], ["en"]), (["ab", "ab"], [["a", "b"]] * 2, ["en"] * 2)):
            with self.assertRaises(ValueError):
                split_corpus(*args)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            split_corpus(["ab", "ab"], [["a", "b"], ["ab"]], ["en", "en"])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            split_corpus(texts, [], languages)
        with self.assertRaises(ValueError):
            load_benchmark_split(languages=(), include_reflective=False)

    def test_singleton_language_is_explicitly_training_only(self):
        split = split_corpus(["ab", "cd", "xy"], [["a", "b"], ["c", "d"], ["x", "y"]], ["en", "en", "es"])
        self.assertEqual(split.receipt["training_only_languages"], ["es"])
        self.assertIn("xy", split.train_texts)

    def test_f1_requires_text_alignment_and_scores_empty_boundary_sets(self):
        self.assertEqual(segmentation_f1("word", ["word"], ["word"]), 1.0)
        self.assertEqual(segmentation_f1("word", ["wo", "rd"], ["word"]), 0.0)
        self.assertEqual(segmentation_f1("", [], []), 1.0)
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            segmentation_f1("a b", ["a ", "b"], ["a", "b"])


class BenchmarkPipelineTest(unittest.TestCase):
    def test_benchmark_generates_metrics_and_reports(self) -> None:
        training_inputs = []
        original_train = OnePassAIT.train_student

        def record_training(instance, texts, segments, **kwargs):
            training_inputs.append(tuple(texts))
            return original_train(instance, texts, segments, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(OnePassAIT, "train_student", record_training):
            metrics = run_benchmark(
                languages=("es",),
                include_reflective=True,
                max_samples=4,
                output_dir=tmpdir,
                seed=101,
            )

            self.assertIn("baseline", metrics)
            self.assertIn("variants", metrics)
            self.assertIn("dataset", metrics)
            baseline = metrics["baseline"]
            split = metrics["dataset"]["split"]
            self.assertEqual(metrics["protocol"]["score_partition"], "held_out_test")
            self.assertEqual(len(training_inputs), 1)
            self.assertFalse(set(training_inputs[0]) & set(baseline["per_text"]))
            self.assertEqual(len(baseline["per_text"]), split["test"]["samples"])
            self.assertEqual(set(hashlib.sha256(text.encode("utf-8")).hexdigest() for text in training_inputs[0]), set(split["train_pool"]["text_sha256"]))
            self.assertIn("f1", baseline)
            latency = baseline["latency_ms"]
            self.assertGreater(latency["p95"], 0.0)
            self.assertLess(latency["p95"], 2000.0)

            variants = metrics["variants"]
            self.assertIn("noise", variants)
            self.assertIn("dialect", variants)
            self.assertIn("tempo_slow", variants)
            self.assertIn("tempo_fast", variants)
            for info in variants.values():
                self.assertEqual(info["samples"], split["test"]["samples"])
            for info in metrics["ablations"].values():
                self.assertEqual(set(info["per_text"]), set(baseline["per_text"]))
                self.assertEqual(info["kind"], "inference_ablation")
                self.assertFalse(info["retrained"])
            self.assertEqual(set(metrics["reference_baselines"]["whitespace_punctuation"]["per_text"]), set(baseline["per_text"]))
            training = metrics["training"]
            if "train" in training:
                train_hashes = set(training["train"]["text_sha256"])
                val_hashes = set(training["validation"]["text_sha256"])
                self.assertFalse(train_hashes & val_hashes)
                self.assertEqual(train_hashes | val_hashes, set(split["train_pool"]["text_sha256"]))

            json_path = os.path.join(tmpdir, "benchmark_report.json")
            md_path = os.path.join(tmpdir, "benchmark_report.md")
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))
            self.assertEqual(json.loads(Path(json_path).read_text())["schema_version"], 2)
            self.assertIn("held-out test", Path(md_path).read_text())

            np_metrics = metrics["np_stub"]
            if np_metrics["available"]:
                self.assertGreaterEqual(np_metrics["linf"], 0.0)
                self.assertGreaterEqual(np_metrics["mse"], 0.0)

    def test_evaluation_script_also_uses_heldout_partition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_evaluation(Path(tmpdir), latency_runs=2, robustness_trials=1, robustness_noise=0.1, seed=101, max_samples=4)
            test_hashes = set(result["split"]["test"]["text_sha256"])
            train_hashes = set(result["split"]["train_pool"]["text_sha256"])
            observed = {hashlib.sha256(row["text"].encode("utf-8")).hexdigest() for row in result["segmentation"]["per_sample"]}
            self.assertEqual(observed, test_hashes)
            self.assertFalse(observed & train_hashes)
            self.assertEqual(len(result["latency_samples_seconds"]), 2)
            self.assertEqual(len(result["robustness"]["records"]), len(test_hashes))
            self.assertTrue((Path(tmpdir) / "evaluation_metrics.json").exists())

    def test_evaluation_rejects_empty_measurements_and_preserves_unicode_length(self):
        for latency_runs, robustness_trials in ((0, 1), (1, 0)):
            with self.assertRaises(ValueError):
                run_evaluation(Path("unused-output"), latency_runs, robustness_trials, 0.1, 1)
        for seed in range(10):
            self.assertEqual(_perturb_text("Straße İ", 0.0, random.Random(seed)), "Straße İ")
            self.assertEqual(len(_perturb_text("Straße İ", 1.0, random.Random(seed))), len("Straße İ"))


if __name__ == "__main__":
    unittest.main()
