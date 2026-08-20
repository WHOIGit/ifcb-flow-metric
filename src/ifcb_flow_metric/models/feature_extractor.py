import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from sklearn.decomposition import PCA

from ifcb_flow_metric.utils.constants import IFCB_ASPECT_RATIO, EDGE_TOLERANCE
from ifcb_flow_metric.utils.dataloader import get_points
from ifcb_flow_metric.utils.feature_config import get_default_feature_config, get_enabled_features


class FeatureExtractor:
    """Compute a wide variety of features for each point cloud."""

    def __init__(
        self, 
        aspect_ratio: float = IFCB_ASPECT_RATIO, 
        edge_tolerance: int = EDGE_TOLERANCE,
        feature_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.aspect_ratio = aspect_ratio
        self.edge_tolerance = edge_tolerance
        self.feature_config = feature_config or get_default_feature_config()
        self.enabled_features = get_enabled_features(self.feature_config)
        
        # Define the complete mapping of features to column names in order
        self.all_feature_names = [
            'mean_x', 'mean_y', 'std_x', 'std_y', 'median_x', 'median_y', 'iqr_x', 'iqr_y',
            'ratio_spread', 'core_fraction',
            'duplicate_fraction', 'max_duplicate_fraction',
            'cv_x', 'cv_y',
            'skew_x', 'skew_y', 'kurt_x', 'kurt_y',
            'angle', 'eigen_ratio',
            'left_edge_fraction', 'right_edge_fraction', 'top_edge_fraction', 'bottom_edge_fraction', 'total_edge_fraction',
            't_y_var',
            'roi_trigger_fraction'
        ]
    
    def get_enabled_feature_names(self) -> List[str]:
        """Get list of enabled feature names in order."""
        return [name for name in self.all_feature_names if self.enabled_features.get(name, True)]

    # ------------------------------------------------------------------
    # Helper feature functions
    # ------------------------------------------------------------------
    def _edge_features(self, points: np.ndarray) -> np.ndarray:
        """Fraction of points within ``edge_tolerance`` of each frame edge."""
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        left = np.sum(points[:, 0] <= x_min + self.edge_tolerance)
        right = np.sum(points[:, 0] >= x_max - self.edge_tolerance)
        top = np.sum(points[:, 1] <= y_min + self.edge_tolerance)
        bottom = np.sum(points[:, 1] >= y_max - self.edge_tolerance)

        counts = np.array([left, right, top, bottom], dtype=float)
        fracs = counts / len(points)
        total = fracs.sum()
        return np.concatenate([fracs, [total]])

    def _duplicate_y_features(self, points: np.ndarray) -> np.ndarray:
        """Fraction of points sharing a ``y`` value and largest duplicate group."""
        _, counts = np.unique(points[:, 1], return_counts=True)
        duplicate_fraction = counts[counts > 1].sum() / len(points)
        max_duplicate_fraction = counts.max() / len(points)
        return np.array([duplicate_fraction, max_duplicate_fraction])

    def _histogram_uniformity(self, values: np.ndarray, bins: int = 10) -> float:
        counts, _ = np.histogram(values, bins=bins)
        mean = counts.mean()
        if mean == 0:
            return 0.0
        return counts.std() / mean

    def _skewness(self, values: np.ndarray) -> float:
        mean = values.mean()
        std = values.std()
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 3)

    def _kurtosis(self, values: np.ndarray) -> float:
        mean = values.mean()
        std = values.std()
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 4) - 3.0

    # ------------------------------------------------------------------
    # Main feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, load_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the configured feature vector for one loaded point cloud.

        :param load_result: dict from ``get_points`` with keys 'pid',
            'points', 'n_total', and 'error'
        :returns: dict with 'pid', 'features' (1-D float64 ndarray in
            :meth:`get_enabled_feature_names` order, or ``None`` if
            extraction failed) and 'error' (human-readable reason when
            features is ``None``, else ``None``)
        """
        pid = load_result.get('pid')
        points = load_result['points']
        if points is None:
            # the loader already recorded why it failed
            return {
                'pid': pid,
                'features': None,
                'error': load_result.get('error') or 'point cloud not loaded',
            }
        if len(points) < 30:
            return {
                'pid': pid,
                'features': None,
                'error': f'distribution has too few points ({len(points)})',
            }
        try:
            # keep a copy for detecting clipped values in the original space
            original_points = points
            # normalise width so x/y roughly comparable
            points = points.copy().astype(float)
            points[:, 0] /= self.aspect_ratio

            # core statistics
            mean = points.mean(axis=0)
            std = points.std(axis=0)
            median = np.median(points, axis=0)
            q1 = np.quantile(points, 0.25, axis=0)
            q3 = np.quantile(points, 0.75, axis=0)
            iqr = q3 - q1

            # ratio of spreads (elongation) and fraction of points in IQR box
            ratio_spread = (iqr[1] + 1e-8) / (iqr[0] + 1e-8)
            in_core = (
                (points[:, 0] >= q1[0])
                & (points[:, 0] <= q3[0])
                & (points[:, 1] >= q1[1])
                & (points[:, 1] <= q3[1])
            )
            core_fraction = in_core.mean()

            # duplicate y values indicate clipping
            dup_features = self._duplicate_y_features(original_points)

            # histogram based uniformity measures
            cv_x = self._histogram_uniformity(points[:, 0])
            cv_y = self._histogram_uniformity(points[:, 1])

            # skew/kurtosis for shape description
            skew_x = self._skewness(points[:, 0])
            skew_y = self._skewness(points[:, 1])
            kurt_x = self._kurtosis(points[:, 0])
            kurt_y = self._kurtosis(points[:, 1])

            # PCA orientation features
            pca = PCA(n_components=2)
            pca.fit(points)
            angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
            eigen_ratio = pca.explained_variance_ratio_[0] / (
                pca.explained_variance_ratio_[1] + 1e-8
            )

            edge_features = self._edge_features(original_points)

            # time features
            y_rolling_mean = pd.Series(points[:, 1]).rolling(window=10).mean()
            t_y_var = y_rolling_mean.var() if not y_rolling_mean.empty else 0.0

            # Trigger detection: fraction of trigger events that yielded a
            # detected (nonzero-area) ROI. Bounded in [0, 1]; carries the
            # same information as a with:without-ROI ratio but without a
            # division-by-zero at 100% detection.
            n_total = load_result.get('n_total') or 0
            roi_trigger_fraction = (len(original_points) / n_total) if n_total > 0 else 0.0

            # Build features list based on enabled features in correct order
            feature_list = []
            
            # Spatial Statistics Features
            if self.enabled_features.get('mean_x', True):
                feature_list.append(mean[0])
            if self.enabled_features.get('mean_y', True):
                feature_list.append(mean[1])
            if self.enabled_features.get('std_x', True):
                feature_list.append(std[0])
            if self.enabled_features.get('std_y', True):
                feature_list.append(std[1])
            if self.enabled_features.get('median_x', True):
                feature_list.append(median[0])
            if self.enabled_features.get('median_y', True):
                feature_list.append(median[1])
            if self.enabled_features.get('iqr_x', True):
                feature_list.append(iqr[0])
            if self.enabled_features.get('iqr_y', True):
                feature_list.append(iqr[1])
            
            # Distribution Shape Features
            if self.enabled_features.get('ratio_spread', True):
                feature_list.append(ratio_spread)
            if self.enabled_features.get('core_fraction', True):
                feature_list.append(core_fraction)
            
            # Clipping Detection Features
            if self.enabled_features.get('duplicate_fraction', True):
                feature_list.append(dup_features[0])
            if self.enabled_features.get('max_duplicate_fraction', True):
                feature_list.append(dup_features[1])
            
            # Histogram Uniformity Features
            if self.enabled_features.get('cv_x', True):
                feature_list.append(cv_x)
            if self.enabled_features.get('cv_y', True):
                feature_list.append(cv_y)
            
            # Statistical Moments Features
            if self.enabled_features.get('skew_x', True):
                feature_list.append(skew_x)
            if self.enabled_features.get('skew_y', True):
                feature_list.append(skew_y)
            if self.enabled_features.get('kurt_x', True):
                feature_list.append(kurt_x)
            if self.enabled_features.get('kurt_y', True):
                feature_list.append(kurt_y)
            
            # PCA Orientation Features
            if self.enabled_features.get('angle', True):
                feature_list.append(angle)
            if self.enabled_features.get('eigen_ratio', True):
                feature_list.append(eigen_ratio)
            
            # Edge Features
            if self.enabled_features.get('left_edge_fraction', True):
                feature_list.append(edge_features[0])
            if self.enabled_features.get('right_edge_fraction', True):
                feature_list.append(edge_features[1])
            if self.enabled_features.get('top_edge_fraction', True):
                feature_list.append(edge_features[2])
            if self.enabled_features.get('bottom_edge_fraction', True):
                feature_list.append(edge_features[3])
            if self.enabled_features.get('total_edge_fraction', True):
                feature_list.append(edge_features[4])
            
            # Temporal Features
            if self.enabled_features.get('t_y_var', True):
                feature_list.append(t_y_var)

            # Trigger Detection Features
            if self.enabled_features.get('roi_trigger_fraction', True):
                feature_list.append(roi_trigger_fraction)

            features = np.array(feature_list)
            return {'pid': pid, 'features': features, 'error': None}
        except Exception as e:
            return {'pid': pid, 'features': None, 'error': f'{type(e).__name__}: {e}'}

    # ------------------------------------------------------------------
    def load_extract(self, pairs: List[Tuple[str, Optional[str]]]) -> List[Dict[str, Any]]:
        """
        Load point clouds and extract features for (pid, adc_path) pairs.
        """
        load_results = [get_points(pid, adc_path) for pid, adc_path in pairs]
        return [self.extract_features(res) for res in load_results]

    def load_extract_parallel(
        self,
        pairs: List[Tuple[str, Optional[str]]],
        chunk_size: int = 100,
        n_jobs: int = -1,
    ) -> List[Dict[str, Any]]:
        """
        Parallel version of :meth:`load_extract` over (pid, adc_path) pairs.
        The pairs are split into chunks of ``chunk_size``; each chunk is
        handled by one parallel worker.
        """
        chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        print(f"Processing {len(pairs)} PIDs in {len(chunks)} chunks of size {chunk_size}")
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.load_extract)(chunk)
            for chunk in tqdm(chunks, desc="Processing chunks")
        )
        flattened: List[Dict[str, Any]] = []
        for chunk_res in results:
            flattened.extend(chunk_res)
        print(f"Processed {len(flattened)} PIDs")
        return flattened
