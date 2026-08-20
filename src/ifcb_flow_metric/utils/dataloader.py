"""
Point cloud loading for IFCB raw data.

``get_points`` takes a bin PID and the path of its ``.adc`` file and returns
the (x, y) point cloud of detected ROIs in that bin.

Directory discovery has two modes:

* No ID file: ``list_adc_paths`` walks the data tree once (via ifcbkit)
  into a pid -> ADC path mapping.
* ID file given: ``get_pid_pairs`` skips the full walk and resolves each
  listed PID with ifcbkit's pruned ``sync_find_fileset``, which descends
  only along that PID's own path. The work per PID is proportional to the
  depth of its path (a handful of small directories), not the size of the
  tree.

``FeatureExtractor.load_extract_parallel`` consumes the resulting
(pid, adc_path) pairs directly.
"""

import numpy as np
from collections import Counter

from ifcbkit import (
    iter_adc_targets,
    SyncIfcbDataDirectory,
    sync_find_fileset,
    DEFAULT_INCLUDE,
    DEFAULT_EXCLUDE,
)

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

    Without an ``id_file`` the data tree is walked once, via
    :func:`list_adc_paths`.

    With an ``id_file`` the full walk is skipped: each listed PID is
    resolved individually by :func:`_find_adc_path`, which calls ifcbkit's
    pruned ``sync_find_fileset`` (descends only into directories whose
    name is a substring of the PID or a known data-dir name). A PID that
    does not resolve to a file gets ``None`` as its path and is surfaced
    as an error by :func:`get_points` (and :func:`summarize_failures`).
    """
    if id_file is None:
        return list(list_adc_paths(data_dir).items())
    with open(id_file) as f:
        pids = [line.strip() for line in f if line.strip()]
    return [(pid, _find_adc_path(data_dir, pid)) for pid in pids]


def _find_adc_path(data_dir, pid):
    """
    Find the path of ``<pid>.adc`` without walking the full tree.

    Delegates to ifcbkit's ``sync_find_fileset``: a pruned search that
    descends only into directories whose name is a substring of the pid
    (or a known data-dir name), so it touches only the directories on
    the path to the pid. ``require_adc=True`` makes it return a basepath
    only when the ``.adc`` is present; ``require_roi=False`` since point
    clouds are built from the ADC alone.

    :returns: the ``.adc`` path, or ``None`` if the pid is not in the tree
    """
    base = sync_find_fileset(
        data_dir, pid,
        include=DEFAULT_INCLUDE,
        exclude=DEFAULT_EXCLUDE,
        require_adc=True,
        require_roi=False,
    )
    return base + '.adc' if base is not None else None


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
        or ``None`` if loading failed), ``'n_total'`` (the number of
        trigger lines in the ADC file, i.e. all triggers including those
        with no detected ROI; 0 when loading failed) and ``'error'`` (a
        reason string when points is ``None``, else ``None``)
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
        # One ADC line per trigger event; blank lines are not triggers.
        # The nonzero-area subset above is the "with ROI" count, so the
        # with/without-ROI split is recoverable by the feature extractor.
        n_total = sum(1 for line in adc_bytes.splitlines() if line.strip())
        return {'pid': pid, 'points': points, 'n_total': n_total, 'error': None}
    except Exception as e:
        return {'pid': pid, 'points': None, 'n_total': 0, 'error': str(e)}


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
