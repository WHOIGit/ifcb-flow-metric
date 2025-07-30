from ifcb import DataDirectory
import numpy as np
from utils.utilities import parallel_map
from tqdm import tqdm

def get_points(pid, directory='.'):
    try:
        dd = DataDirectory(directory)

        sample_bin = dd[pid]
        cols = sample_bin.schema
        adc = sample_bin.adc

        points = np.vstack(
            [adc[cols.ROI_X], adc[cols.ROI_Y]]
        ).T.astype(np.float64)

        if pid.startswith('I'):
            t_col = cols.FRAME_GRAB_TIME
        else:
            t_col = cols.GRAB_TIME_START

        t = adc[t_col]
    
        return { 'pid': pid, 'points': points, 't': t }
    
    except Exception as e:

        return { 'pid': pid, 'points': None, 't': None, 'error': str(e) }


def get_points_parallel(pids, directory='.', n_jobs=-1):
    return parallel_map(
        get_points,
        pids,
        lambda x: (x, directory),
        n_jobs=n_jobs
    )


