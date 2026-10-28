"""The resource probe must select its implementation before package imports."""
from pathlib import Path
import subprocess
import sys


PROBE = Path(__file__).resolve().parents[1] / "scripts/probe_observation_transactions.py"


def test_importing_probe_does_not_import_numpy_or_ait():
    command = ("import importlib.util,sys; "
               f"spec=importlib.util.spec_from_file_location('transaction_probe',{str(PROBE)!r}); "
               "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
               "assert 'numpy' not in sys.modules; "
               "assert not any(k.startswith('spiralreality_AIT_onepass_aifcore_integrated') for k in sys.modules)")
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_existing_probe_receipt_is_preserved_before_loading_any_package(tmp_path):
    output = tmp_path / "receipt.json"
    output.write_text("preserved original receipt")
    result = subprocess.run([sys.executable, str(PROBE), "--package-root", str(tmp_path),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
    assert output.read_text() == "preserved original receipt"


def test_missing_selected_source_cannot_fall_back_to_current_installation(tmp_path):
    output = tmp_path / "receipt.json"
    result = subprocess.run([sys.executable, str(PROBE), "--package-root", str(tmp_path),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "package-root must contain" in result.stderr
    assert not output.exists()
