"""
Tests for FeatureExtractor.load_extract and load_extract_parallel, the
(pairs-in, feature-rows-out) entry points used by train.py and score.py.
Uses synthetic ADC files (no real IFCB data required).
"""

import numpy as np
import pytest

from ifcb_flow_metric.models.feature_extractor import FeatureExtractor

V2_PID = 'D20130526T095207_IFCB013'


def v2_line(x, y, w, h, trigger=1):
    """One D-style ADC line: 18 fields, x/y/w/h at 13/14/15/16, offset at 17."""
    fields = ['0'] * 18
    fields[0] = str(trigger)
    fields[1] = '0.5'
    fields[13], fields[14], fields[15], fields[16] = str(x), str(y), str(w), str(h)
    fields[17] = '1000'
    return ','.join(fields)


def write_adc(tmp_path, pid, n_lines, vary=False):
    """Write an ADC with ``n_lines`` valid ROI-bearing trigger lines."""
    lines = []
    for i in range(1, n_lines + 1):
        # vary the positions so different PIDs get different clouds
        x = i if vary else 100
        y = i * 3 if vary else 200
        lines.append(v2_line(x, y, 50, 30))
    path = tmp_path / (pid + '.adc')
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def test_load_extract_happy_path(tmp_path):
    path = write_adc(tmp_path, V2_PID, 40)
    results = FeatureExtractor().load_extract([(V2_PID, path)])
    assert len(results) == 1
    result = results[0]
    assert result['pid'] == V2_PID
    assert result['error'] is None
    assert result['features'] is not None
    assert len(result['features']) == len(
        FeatureExtractor().get_enabled_feature_names())


def test_load_extract_preserves_order_and_surfaces_errors(tmp_path):
    ok_pid = 'D20130526T095207_IFCB013'
    short_pid = 'D20130526T095208_IFCB013'
    ok_path = write_adc(tmp_path, ok_pid, 40)
    short_path = write_adc(tmp_path, short_pid, 5)  # < 30 points

    pairs = [
        (ok_pid, ok_path),
        ('D20130526T095209_IFCB013', None),  # absent from the tree
        (short_pid, short_path),
    ]
    results = FeatureExtractor().load_extract(pairs)

    # results come back in input order, one per pair
    assert [r['pid'] for r in results] == [pid for pid, _ in pairs]
    assert results[0]['error'] is None
    assert results[0]['features'] is not None
    assert results[1]['features'] is None
    assert f'no .adc file found' in results[1]['error']
    assert results[2]['features'] is None
    assert 'too few points' in results[2]['error']


def test_load_extract_empty_pairs():
    assert FeatureExtractor().load_extract([]) == []


def test_load_extract_parallel_matches_serial(tmp_path):
    pids = [f'D20130526T0952{minute:02d}_IFCB013' for minute in range(7, 12)]
    pairs = [(pid, write_adc(tmp_path, pid, 40, vary=True)) for pid in pids]

    serial = FeatureExtractor().load_extract(pairs)
    parallel = FeatureExtractor().load_extract_parallel(pairs, chunk_size=2)

    assert [r['pid'] for r in parallel] == [r['pid'] for r in serial]
    for par, ser in zip(parallel, serial):
        assert par['error'] == ser['error'] is None
        assert np.array_equal(par['features'], ser['features'])


def test_load_extract_parallel_mixed_failures(tmp_path):
    ok_pid = 'D20130526T095207_IFCB013'
    short_pid = 'D20130526T095208_IFCB013'
    pairs = [
        (ok_pid, write_adc(tmp_path, ok_pid, 40)),
        ('D20130526T095209_IFCB013', None),
        (short_pid, write_adc(tmp_path, short_pid, 5)),
    ]
    results = FeatureExtractor().load_extract_parallel(pairs, chunk_size=2)
    assert [r['pid'] for r in results] == [pid for pid, _ in pairs]
    assert results[0]['features'] is not None
    assert results[1]['features'] is None
    assert 'too few points' in results[2]['error']


def test_load_extract_parallel_chunks_pairs(tmp_path, monkeypatch):
    # 5 pairs with chunk_size=2 -> exactly 3 load_extract calls of
    # sizes 2, 2, 1, in order (n_jobs=1 keeps it in-process so the
    # instance-level patch below is observed)
    pids = [f'D20130526T0952{minute:02d}_IFCB013' for minute in range(12, 17)]
    pairs = [(pid, write_adc(tmp_path, pid, 40)) for pid in pids]

    extractor = FeatureExtractor()
    original = extractor.load_extract
    chunk_sizes = []
    def counting(chunk):
        chunk_sizes.append(len(chunk))
        return original(chunk)
    extractor.load_extract = counting

    results = extractor.load_extract_parallel(pairs, chunk_size=2, n_jobs=1)

    assert chunk_sizes == [2, 2, 1]
    assert [r['pid'] for r in results] == [pid for pid, _ in pairs]


def test_load_extract_parallel_empty_pairs():
    assert FeatureExtractor().load_extract_parallel([]) == []
