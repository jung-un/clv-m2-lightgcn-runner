"""Exercise the actual Colab setup cell before it clones or trains anything."""
import sys
import types

import nbformat
import pytest


def _setup_prefix(tmp_path, monkeypatch, mount):
    notebook = nbformat.read('clv_linear_nv_original_m4_colab.ipynb', as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.get('id') == 'setup')
    source = source.split("SOURCE_COMMIT=", 1)[0]
    source = source.replace('/content/drive/MyDrive/논문/data', str(tmp_path))
    colab = types.ModuleType('google.colab')
    colab.drive = types.SimpleNamespace(mount=mount)
    google = types.ModuleType('google')
    google.colab = colab
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    return source, tmp_path / 'results_v3_dunnhumby_history_linear_nv_es_v2/reports/30287cad8e5ed1d1/result.json'


def test_existing_report_skips_mount(tmp_path, monkeypatch):
    def mount(_):
        pytest.fail('mounted Drive should not be mounted again')
    source, report = _setup_prefix(tmp_path, monkeypatch, mount)
    report.parent.mkdir(parents=True)
    report.write_text('{}')
    exec(compile(source, '<colab setup>', 'exec'), {})


def test_mount_failure_stops_before_clone_or_training(tmp_path, monkeypatch):
    def mount(_):
        raise ValueError('mount failed')
    source, _ = _setup_prefix(tmp_path, monkeypatch, mount)
    monkeypatch.setattr('os.path.ismount', lambda _: False)
    with pytest.raises(RuntimeError, match='Drive 연결이 실패') as error:
        exec(compile(source, '<colab setup>', 'exec'), {})
    assert isinstance(error.value.__cause__, ValueError)


def test_mount_succeeds_only_when_required_report_is_readable(tmp_path, monkeypatch):
    report_holder = {}
    def mount(_):
        report = report_holder['report']
        report.parent.mkdir(parents=True)
        report.write_text('{}')
    source, report = _setup_prefix(tmp_path, monkeypatch, mount)
    report_holder['report'] = report
    monkeypatch.setattr('os.path.ismount', lambda _: False)
    exec(compile(source, '<colab setup>', 'exec'), {})
