import numpy as np

from ifcb import DataDirectory

IFCB_ASPECT_RATIO = 1.36

class AdcLoader(object):

    def __init__(self, directory='.'):
        self.dd = DataDirectory(directory)

    def _get_points(self, pid):
        sample_bin = self.dd[pid]
        cols = sample_bin.schema
        adc = sample_bin.adc
        return np.vstack(
            [adc[cols.ROI_X], adc[cols.ROI_Y]]
        ).T.astype(np.float64)
    
    def __getitem__(self, pid):
        return self._get_points(pid)
    
    def __iter__(self):
        for sample_bin in self.dd:
            yield sample_bin.lid
        