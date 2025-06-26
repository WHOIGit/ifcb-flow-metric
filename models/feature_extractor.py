import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Any, Dict, List
from sklearn.decomposition import PCA

from utils.constants import IFCB_ASPECT_RATIO, EDGE_TOLERANCE
from utils.dataloader import get_points


class FeatureExtractor:
    """Compute a wide variety of features for each point cloud."""

    def __init__(self, aspect_ratio: float = IFCB_ASPECT_RATIO, edge_tolerance: int = EDGE_TOLERANCE) -> None:
        self.aspect_ratio = aspect_ratio
        self.edge_tolerance = edge_tolerance

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

            features = np.concatenate(
                [
                    mean,
                    std,
                    median,
                    iqr,
                    [ratio_spread, core_fraction],
                    dup_features,
                    [cv_x, cv_y],
                    [skew_x, skew_y, kurt_x, kurt_y],
                    [angle, eigen_ratio],
                    edge_features,
                ]
            )
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
