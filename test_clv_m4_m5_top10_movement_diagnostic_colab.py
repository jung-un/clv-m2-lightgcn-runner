"""Check the read-only Colab setup before Drive data loading."""
from pathlib import Path
import sys
import types

import nbformat
import pytest


NOTEBOOK = 'clv_m4_m5_top10_movement_diagnostic_colab.ipynb'


def _setup_prefix(tmp_path, monkeypatch, mount):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.get('id') == 'setup')
    source = source.split('SOURCE_COMMIT =', 1)[0]
    source = source.replace('/content/drive/MyDrive/논문/data', str(tmp_path))
    colab = types.ModuleType('google.colab')
    colab.drive = types.SimpleNamespace(mount=mount)
    google = types.ModuleType('google')
    google.colab = colab
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    report = tmp_path/'results_v3_dunnhumby_m5_linear_nv_original_m4_lambda025_seed43_v1/reports/result.json'
    return source, report


def test_schema_syntax_and_read_only_commit_pin():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, f'notebook/{cell.id}', 'exec')
    source = next(cell.source for cell in notebook.cells if cell.get('id') == 'setup')
    assert "SOURCE_COMMIT = 'cea61be1f7844a5bcf98fd0ea7d8a94945a0b9d5'" in source
    assert Path('clv_m4_m5_top10_movement_diagnostic.py').is_file()
    assert all('screen.run(' not in cell.source for cell in notebook.cells)


def test_existing_report_skips_drive_remount(tmp_path, monkeypatch):
    source, report = _setup_prefix(tmp_path, monkeypatch, lambda _: pytest.fail('no remount'))
    report.parent.mkdir(parents=True)
    report.write_text('{}', encoding='utf-8')
    exec(compile(source, '<setup>', 'exec'), {})


def test_mount_failure_stops_without_clone(tmp_path, monkeypatch):
    def mount(_):
        raise ValueError('mount failed')
    source, _ = _setup_prefix(tmp_path, monkeypatch, mount)
    monkeypatch.setattr('os.path.ismount', lambda _: False)
    with pytest.raises(RuntimeError, match='Drive 연결 실패') as exc:
        exec(compile(source, '<setup>', 'exec'), {})
    assert isinstance(exc.value.__cause__, ValueError)
