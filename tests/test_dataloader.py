"""
Tests for point cloud loading, error surfacing, and directory discovery
(full-tree walk vs pruned per-PID search), using synthetic ADC files
(no real IFCB data required).
"""

import os

import numpy as np
import pytest

from ifcbkit import parse_pid

from ifcb_flow_metric.models.feature_extractor import FeatureExtractor
from ifcb_flow_metric.models.inference import Inferencer
from ifcb_flow_metric.utils.dataloader import (
    get_points,
    get_pid_pairs,
    list_adc_paths,
    summarize_failures,
)

V2_PID = 'D20130526T095207_IFCB013'
V1_PID = 'IFCB5_2012_028_081515'


def v2_line(x, y, w, h, trigger=1, corrupt=False):
    """One D-style ADC line: 18 fields, x/y/w/h at 13/14/15/16, offset at 17."""
    fields = ['0'] * 18
    fields[0] = str(trigger)
    fields[1] = '0.5'
    fields[13], fields[14], fields[15], fields[16] = str(x), str(y), str(w), str(h)
    fields[17] = '1000'
    if corrupt:
        fields[13] = 'notanint'
    return ','.join(fields)


def v1_line(x, y, w, h, trigger=1):
    """One I-style ADC line: 14 fields, x/y/w/h at 9/10/11/12, offset at 13."""
    fields = ['0'] * 14
    fields[0] = str(trigger)
    fields[1] = '0.5'
    fields[9], fields[10], fields[11], fields[12] = str(x), str(y), str(w), str(h)
    fields[13] = '1000'
    return ','.join(fields)


def write_adc(tmp_path, pid, lines):
    path = tmp_path / (pid + '.adc')
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def test_get_points_v2_drops_zero_area_and_malformed_lines(tmp_path):
    path = write_adc(tmp_path, V2_PID, [
        v2_line(100, 200, 50, 30),
        v2_line(0, 0, 0, 0),              # no-ROI trigger
        v2_line(150, 250, 60, 40),
        v2_line(1, 2, 3, 4, corrupt=True),  # malformed line
    ])
    result = get_points(V2_PID, path)
    assert result['error'] is None
    assert result['pid'] == V2_PID
    assert result['points'].tolist() == [[100.0, 200.0], [150.0, 250.0]]


def test_get_points_keeps_zero_coordinate_roi_with_area(tmp_path):
    # a real detection at the origin is NOT a no-ROI sentinel: it has area
    path = write_adc(tmp_path, V2_PID, [v2_line(0, 0, 40, 25)])
    result = get_points(V2_PID, path)
    assert result['error'] is None
    assert result['points'].tolist() == [[0.0, 0.0]]


def test_get_points_v1_column_layout(tmp_path):
    path = write_adc(tmp_path, V1_PID, [
        v1_line(10, 20, 30, 40),
        v1_line(11, 21, 31, 41),
    ])
    result = get_points(V1_PID, path)
    assert result['error'] is None
    assert result['points'].tolist() == [[10.0, 20.0], [11.0, 21.0]]


def test_get_points_empty_adc(tmp_path):
    path = write_adc(tmp_path, V2_PID, [])
    result = get_points(V2_PID, path)
    assert result['error'] is None
    assert result['points'].shape == (0, 2)


def test_get_points_missing_file():
    result = get_points(V2_PID, '/nonexistent/nope.adc')
    assert result['points'] is None
    assert 'nope.adc' in result['error']


def test_get_points_none_path():
    result = get_points(V2_PID, None)
    assert result['points'] is None
    assert result['error'] == f'no .adc file found for {V2_PID}'


def test_extract_features_propagates_loader_error(tmp_path):
    path = write_adc(tmp_path, V2_PID, [])  # 0 points -> too few
    result = FeatureExtractor().extract_features(get_points(V2_PID, path))
    assert result['features'] is None
    assert 'too few points' in result['error']


def test_extract_features_reports_extraction_failure():
    # NaN in the cloud (from a partially unparseable file) must be
    # surfaced as an error, not silently dropped or crash the batch
    points = np.zeros((31, 2))
    points[5, 0] = np.nan
    result = FeatureExtractor().extract_features(
        {'pid': V2_PID, 'points': points, 'error': None})
    assert result['features'] is None
    assert result['error'] is not None


def test_extract_features_success_has_no_error():
    points = np.random.default_rng(42).integers(0, 1000, size=(40, 2)).astype(float)
    result = FeatureExtractor().extract_features(
        {'pid': V2_PID, 'points': points, 'error': None})
    assert result['error'] is None
    assert result['features'] is not None
    assert len(result['features']) == len(
        FeatureExtractor().get_enabled_feature_names())


def test_inferencer_all_pids_failed(tmp_path):
    # scoring a batch in which every PID failed must not crash; it must
    # return NaN rows only (this used to raise a 1-D-array ValueError)
    path = write_adc(tmp_path, V2_PID, [])  # 0 points -> too few
    feature_results = [FeatureExtractor().extract_features(get_points(V2_PID, path))]
    scores = Inferencer(model=object()).score_distributions(feature_results)
    assert len(scores) == 1
    assert scores[0]['pid'] == V2_PID
    assert np.isnan(scores[0]['anomaly_score'])


def _make_fileset(root, pid, n_lines=1):
    # standard IFCB layout: D-style nests data/{year}/{Dyyyymm}/{Dyyyymmdd},
    # I-style nests data/{year}/{IFCBn_yyyy_ddd}
    parsed = parse_pid(pid)
    day_dir = parsed['day_dir']
    if day_dir.startswith('D'):
        parts = [str(parsed['year']), day_dir[:-2], day_dir]
    else:
        parts = [str(parsed['year']), day_dir]
    day = root.joinpath(*(['data'] + parts))
    day.mkdir(parents=True, exist_ok=True)
    (day / (pid + '.hdr')).write_text('ifcb5:\n  instrumentName: IFCB13\n')
    (day / (pid + '.adc')).write_text('\n'.join(
        [v2_line(i, i, 5, 5) for i in range(1, n_lines + 1)]) + '\n')


def test_list_adc_paths_single_pass(tmp_path):
    _make_fileset(tmp_path, V2_PID)
    expected = str(tmp_path / 'data' / '2013' / 'D201305' / 'D20130526' / (V2_PID + '.adc'))
    assert list_adc_paths(str(tmp_path)) == {V2_PID: expected}


def test_get_pid_pairs_all_and_filtered(tmp_path):
    _make_fileset(tmp_path, V2_PID)

    # no ID file: everything in the tree
    pairs = dict(get_pid_pairs(str(tmp_path)))
    assert set(pairs) == {V2_PID}

    # ID file: listed PID resolves, unknown PID gets a None path
    id_file = tmp_path / 'ids.txt'
    unknown = 'D19990101T000000_IFCB999'
    id_file.write_text(f'{V2_PID}\n{unknown}\n\n')
    pairs = get_pid_pairs(str(tmp_path), str(id_file))
    assert [pid for pid, _ in pairs] == [V2_PID, unknown]
    assert pairs[0][1].endswith(V2_PID + '.adc')
    assert pairs[1][1] is None


def test_get_pid_pairs_id_file_skips_full_tree_walk(tmp_path, monkeypatch):
    # two days in one year; the ID file only covers the second day.
    # the pruned search must list only the directories leading to that
    # day, never the sibling day directory.
    other_day_pid = 'D20130527T100000_IFCB013'
    _make_fileset(tmp_path, V2_PID)              # day dir D20130526
    _make_fileset(tmp_path, other_day_pid)      # day dir D20130527

    id_file = tmp_path / 'ids.txt'
    id_file.write_text(other_day_pid + '\n')

    calls = []
    real_listdir = os.listdir
    def counting_listdir(path):
        calls.append(path)
        return real_listdir(path)
    monkeypatch.setattr('os.listdir', counting_listdir)

    pairs = get_pid_pairs(str(tmp_path), str(id_file))
    assert pairs[0][1].endswith(other_day_pid + '.adc')
    assert not any(c.endswith('D20130526') for c in calls), (
        f'sibling day dir was listed: {calls}')
    assert len(calls) <= 5  # root, data/, 2013/, D201305/, the one day dir


def test_summarize_failures(capsys):
    results = [
        {'pid': 'a', 'features': None, 'error': 'no .adc file found for a'},
        {'pid': 'b', 'features': np.zeros(3), 'error': None},
    ]
    summarize_failures(results)
    out = capsys.readouterr().out
    assert 'a' in out and 'no .adc file found for a' in out
    assert 'b' not in out
