import json
from clv_existing_m4_readout import readout


def fixture(tmp_path, rows, summary=True):
    folder = tmp_path / 'results_v3_hm_m4_k1_assignment_control_hm2y_development_screen_v1'
    folder.mkdir(exist_ok=True)
    if summary:
        (folder / 'report_abc.json').write_text(json.dumps({
            'config': {'epochs': 100, 'seed': 42}, 'absolute_rows': rows}))
    else:
        arm = folder / 'arms' / 'abc'
        arm.mkdir(parents=True)
        for row in rows:
            (arm / (row['model_id'] + '.json')).write_text(json.dumps(row))
    return folder


def test_partial_results_no_training(tmp_path):
    fixture(tmp_path, [{'model_id': 'm1_bpr', 'seed': 42, 'metrics': {'recall@10': .1}},
                       {'model_id': 'm4_actual', 'seed': 42, 'metrics': {'recall@10': .11}}], False)
    tables = readout(tmp_path)
    assert len(tables['absolute']) == 2
    assert round(tables['within_run_comparison'].iloc[0].relative_change_pct) == 10
    assert tables['hm_presence'].query('seed == 43').completed_result_count.sum() == 0


def test_parent_and_arm_merge(tmp_path):
    folder = fixture(tmp_path, [{'model_id': 'm1_bpr', 'seed': 42, 'recall@10': .1}])
    arm = folder / 'arms' / 'abc'
    arm.mkdir(parents=True)
    (arm / 'm1.json').write_text(json.dumps({'model_id': 'm1_bpr', 'seed': 42, 'metrics': {'recall@10': .1}}))
    tables = readout(tmp_path)
    assert len(tables['absolute']) == 1
    assert '설정 있음' in tables['inventory'].iloc[0].metadata_status


def test_conflicting_duplicate_cannot_compare(tmp_path):
    folder = fixture(tmp_path, [{'model_id': 'm1_bpr', 'seed': 42, 'recall@10': .1},
                                {'model_id': 'm4_actual', 'seed': 42, 'recall@10': .11}])
    arm = folder / 'arms' / 'abc'
    arm.mkdir(parents=True)
    (arm / 'm4.json').write_text(json.dumps({'model_id': 'm4_actual', 'seed': 42, 'metrics': {'recall@10': .12}}))
    tables = readout(tmp_path)
    assert not tables['warnings'].empty
    assert tables['within_run_comparison'].empty


def test_zero_denominator_and_no_cross_seed_pairing(tmp_path):
    fixture(tmp_path, [{'model_id': 'm1_bpr', 'seed': 42, 'recall@10': 0},
                       {'model_id': 'm4_actual', 'seed': 43, 'recall@10': .1},
                       {'model_id': 'm4_actual', 'seed': 42, 'recall@10': .1}])
    table = readout(tmp_path)['within_run_comparison']
    assert len(table) == 1
    assert table.relative_change_pct.isna().all()


def test_checkpoint_is_not_deserialized(tmp_path):
    folder = fixture(tmp_path, [])
    (folder / 'unfinished.pt').write_bytes(b'not a torch checkpoint')
    tables = readout(tmp_path)
    assert len(tables['checkpoints']) == 1
    assert tables['absolute'].empty
