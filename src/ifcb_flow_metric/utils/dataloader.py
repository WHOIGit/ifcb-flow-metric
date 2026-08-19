"""
Point cloud loading for IFCB raw data.

``get_points`` takes a bin PID and the path of its ``.adc`` file and returns
the (x, y) point cloud of detected ROIs in that bin.

Directory discovery is a single pass: ``list_adc_paths`` walks the data tree
once (via ifcbkit) into a pid -> ADC path mapping, and ``get_pid_pairs``
combines that with an optional ID file. ``FeatureExtractor.load_extract_parallel``
consumes the resulting (pid, adc_path) pairs directly, so there is no
per-bin recursive directory search.
"""

import numpy as np
from collections import Counter

from ifcbkit import iter_adc_targets, SyncIfcbDataDirectory

from ifcb_flow_metric.utils.utilities import parallel_map


def list_adc_paths(directory):
    """
    Walk the IFCB data tree once, returning a mapping of bin PID to the
    path of its ``.adc`` file.

    Only filesets with an ``.adc`` file are included. ``.roi`` files are not
    required, since point clouds are built from the ADC alone.
    """
    dd = SyncIfcbDataDirectory(directory, require_adc=True, require_roi=False)
    return {fileset['pid']: fileset['adc'] for fileset in dd.list()}


def get_pid_pairs(data_dir, id_file=None):
    """
    Return a list of ``(pid, adc_path)`` pairs to load.

    The data tree is walked once, via :func:`list_adc_paths`. If
    ``id_file`` is given, the result is restricted to the PIDs listed
    in it (one per line); a PID that does not resolve to a file in the
    tree gets ``None`` as its path and is surfaced as an error by
    :func:`get_points`.
    """
    adc_paths = list_adc_paths(data_dir)
    if id_file is None:
        return list(adc_paths.items())
    with open(id_file) as f:
        pids = [line.strip() for line in f if line.strip()]
    return [(pid, adc_paths.get(pid)) for pid in pids]


def get_points(pid, adc_path):
    """
    Load the (x, y) point cloud of detected ROIs for a single bin.

    Only triggers with a detected (nonzero-area) ROI contribute a point;
    the ifcbkit parser drops zero-area and malformed lines. Points are in
    ADC file order.

    :param pid: the bin PID, e.g. ``'D20221227T093138_IFCB127'``; selects
        the ADC column layout (I-style and D-style differ)
    :param adc_path: path to the bin's ``.adc`` file. May be ``None``
        (e.g. for an ID-file PID absent from the tree), in which case a
        descriptive error is returned.
    :returns: dict with ``'pid'``, ``'points'`` (an (N, 2) float64 ndarray,
        or ``None`` if loading failed) and ``'error'`` (a reason string
        when points is ``None``, else ``None``)
    """
    try:
        if adc_path is None:
            raise ValueError(f'no .adc file found for {pid}')
        with open(adc_path, 'rb') as f:
            adc_bytes = f.read()
        points = np.array(
            [(record['x'], record['y']) for record in iter_adc_targets(pid, adc_bytes)],
            dtype=np.float64,
        ).reshape(-1, 2)
        return {'pid': pid, 'points': points, 'error': None}
    except Exception as e:
        return {'pid': pid, 'points': None, 'error': str(e)}


def get_points_parallel(pairs, n_jobs=-1):
    """
    Map :func:`get_points` over ``(pid, adc_path)`` pairs in parallel.
    """
    return parallel_map(get_points, pairs, lambda pair: pair, n_jobs=n_jobs)


def summarize_failures(feature_results, verbose_limit=20, limit=10):
    """
    Print the PIDs that failed to produce features, with the reason each
    failed. Prints one line per PID when there are few failures, or a
    count grouped by error message when there are many.
    """
    failed = [r for r in feature_results if r['features'] is None]
    if not failed:
        return
    print(f'\n{len(failed)} PIDs failed to produce features:')
    if len(failed) <= verbose_limit:
        for r in failed:
            print(f"  {r['pid']}: {r['error']}")
        return
    for error, count in Counter(r['error'] for r in failed).most_common(limit):
        print(f'  {count:5d}x  {error}')
