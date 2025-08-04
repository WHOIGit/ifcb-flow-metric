import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from sklearn.decomposition import PCA

from utils.constants import IFCB_ASPECT_RATIO, EDGE_TOLERANCE
from utils.dataloader import get_points
from utils.feature_config import get_default_feature_config, get_enabled_features


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
            'second_t_value', 't_var'
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
        try:
            pid = load_result["pid"]
            points = load_result["points"]
            t = load_result["t"]
            if points is None or len(points) < 30:
                raise ValueError("Distribution has too few points")

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
            second_t_value = t[1] if len(t) > 1 else t[0]
            t_var = np.var(np.diff(t)) if len(t) > 1 else 0.0

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
            if self.enabled_features.get('second_t_value', True):
                feature_list.append(second_t_value)
            if self.enabled_features.get('t_var', True):
                feature_list.append(t_var)
            
            features = np.array(feature_list)
            return {"pid": pid, "features": features}
        except Exception:
            return {"pid": load_result.get("pid"), "features": None}

    # ------------------------------------------------------------------
    def load_extract(self, pids: List[str], directory: str) -> List[Dict[str, Any]]:
        load_results = [get_points(pid, directory) for pid in pids]
        feature_results = [self.extract_features(res) for res in load_results]
        return feature_results

    def load_extract_parallel(
        self,
        pids: List[str],
        directory: str,
        chunk_size: int = 100,
        n_jobs: int = -1,
    ) -> List[Dict[str, Any]]:
        chunks = [pids[i : i + chunk_size] for i in range(0, len(pids), chunk_size)]
        print(f"Processing {len(pids)} PIDs in {len(chunks)} chunks of size {chunk_size}")
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.load_extract)(chunk, directory)
            for chunk in tqdm(chunks, desc="Processing chunks")
        )
        flattened: List[Dict[str, Any]] = []
        for chunk_res in results:
            flattened.extend(chunk_res)
        print(f"Processed {len(flattened)} PIDs successfully")
        return flattened
