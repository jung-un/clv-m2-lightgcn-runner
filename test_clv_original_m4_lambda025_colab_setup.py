"""Check the notebook's real setup cell before clone or GPU training."""
import sys
import types

import nbformat
import pytest


def setup_prefix(tmp_path, monkeypatch, mount):
    notebook = nbformat.read('clv_original_m4_lambda025_colab.ipynb', as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.get('id') == 'setup')
    source = source.split('SOURCE_COMMIT =', 1)[0]
    source = source.replace('/content/drive/MyDrive/논문/data', str(tmp_path))
    colab = types.ModuleType('google.colab')
    colab.drive = types.SimpleNamespace(mount=mount)
    google = types.ModuleType('google')
    google.colab = colab
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    report = tmp_path/'results_v3_dunnhumby_linear_nv_original_m4_validity_masked_seed43_v2/reports/result.json'
    return source, report


def test_existing_report_skips_mount(tmp_path, monkeypatch):
    source, report = setup_prefix(tmp_path, monkeypatch, lambda _: pytest.fail('no remount'))
    report.parent.mkdir(parents=True)
    report.write_text('{}')
    exec(compile(source, '<notebook setup>', 'exec'), {})


def test_mount_failure_stops_before_clone(tmp_path, monkeypatch):
    def mount(_):
        raise ValueError('mount failed')
    source, _ = setup_prefix(tmp_path, monkeypatch, mount)
    monkeypatch.setattr('os.path.ismount', lambda _: False)
    with pytest.raises(RuntimeError, match='Drive 연결 실패') as exc:
        exec(compile(source, '<notebook setup>', 'exec'), {})
    assert isinstance(exc.value.__cause__, ValueError)


def test_mount_succeeds_only_if_report_is_visible(tmp_path, monkeypatch):
    holder = {}
    def mount(_):
        report = holder['report']
        report.parent.mkdir(parents=True)
        report.write_text('{}')
    source, report = setup_prefix(tmp_path, monkeypatch, mount)
    holder['report'] = report
    monkeypatch.setattr('os.path.ismount', lambda _: False)
    exec(compile(source, '<notebook setup>', 'exec'), {})
