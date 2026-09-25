"""Check the pinned Colab notebook before any clone, Drive data load or training."""
from pathlib import Path
import sys
import types

import nbformat
import pytest


NOTEBOOK = 'clv_m5_linear_nv_original_m4_lambda025_colab.ipynb'


def setup_prefix(tmp_path, monkeypatch, mount):
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
    report = tmp_path/'results_v3_dunnhumby_original_m4_lambda025_seed43_v1/reports/result.json'
    return source, report


def test_notebook_schema_code_and_pinned_commit():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, f"notebook/{cell.id}", 'exec')
    source = next(cell.source for cell in notebook.cells if cell.get('id') == 'setup')
    assert "SOURCE_COMMIT = '1697af8f980dd9000740b2beaca7ef28d26f94fd'" in source
    assert 'clv_m5_linear_nv_original_m4_lambda025_screen' in source
    assert Path('clv_m5_linear_nv_original_m4_lambda025_screen.py').is_file()


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
