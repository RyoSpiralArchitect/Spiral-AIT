import importlib.util
import json
from pathlib import Path

from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import ExternalSentence
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def test_trilingual_evaluation_runs_without_network_and_preserves_text():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("trilingual_runner", root / "scripts/benchmark_trilingual.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = json.loads((root / "configs/trilingual-context-pilot.json").read_text())
    class Corpus:
        group_for_document = {"demo": "demo"}
    examples = {"en": ("Test", " ", "input", "."), "ja": ("日本", "語", "です", "。"), "zh": ("中文", "测试", "。")}
    rows = [ExternalSentence(lang, "demo", lang, lang, "".join(parts), parts) for lang, parts in examples.items()]
    model = OnePassAIT(latent_dim=8, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")
    temperatures = runner.fit_development_temperatures(model, rows, [1, 2])
    results = runner.evaluate(model, rows, Corpus(), config, temperatures)
    assert len(results) == 3
    for row, result in zip(rows, results):
        assert result["stream_raw"]["boundary"]["positions"] == len(row.text) - 1
        assert result["stream_calibrated"]["boundary"]["positions"] == len(row.text) - 1
        assert result["confidence_raw"]["positions"] == len(row.text) - 1
