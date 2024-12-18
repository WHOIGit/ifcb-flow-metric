import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class DistributionSeriesClassifier:
    """
    Analyzes a series of point cloud distributions to identify which ones match
    the dominant pattern in the dataset. Accounts for camera aspect ratio.
    """
    
    def __init__(self, aspect_ratio=1.0, contamination=0.1):
        """
        Initialize classifier.
        
        Parameters:
        aspect_ratio: float, width/height ratio of the camera frame
        contamination: float, expected fraction of anomalous distributions
        """
        self.aspect_ratio = aspect_ratio
        self.contamination = contamination
        self.fitted = False
    
    def _normalize_points(self, points):
        """
        Normalize points to account for aspect ratio by scaling x coordinates.
        """
        normalized = points.copy()
        normalized[:, 0] = normalized[:, 0] / self.aspect_ratio
        return normalized
    
    def _extract_distribution_features(self, points):
        """Extract statistical features from a single point cloud distribution."""
        # Some features won't converge or be useful if there are too few points
        if points.shape[0] < 30:  # 30 because 20 is the minimum for LOF
            raise ValueError("Distribution has too few points")

        # Normalize points to account for aspect ratio
        normalized_points = self._normalize_points(points)

        # Single component GMM features
        gmm = GaussianMixture(n_components=1, random_state=42)
        gmm.fit(normalized_points)
        
        means = gmm.means_[0]  # Single component mean
        covs = gmm.covariances_[0]  # Single component covariance
        
        # Basic stats on normalized points
        center = np.mean(normalized_points, axis=0)
        spread = np.std(normalized_points, axis=0)
        
        # LOF features on normalized points
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(normalized_points)
        lof_scores = -lof.negative_outlier_factor_
        
        # PCA features on normalized points
        pca = PCA(n_components=2)
        pca.fit(normalized_points)
        
        # First component angle (in radians)
        first_component = pca.components_[0]
        angle = np.arctan2(first_component[1], first_component[0])
        
        # Ratio of eigenvalues (indicates shape elongation)
        eigenvalue_ratio = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]
        
        # Percent variance explained by first component
        variance_explained = pca.explained_variance_ratio_[0]
        
        # Combine all features
        features = np.concatenate([
            means.flatten(),  # 2 features
            covs.flatten(),   # 4 features (2x2 symmetric matrix)
            center,          # 2 features
            spread,          # 2 features
            [np.mean(lof_scores), np.std(lof_scores)],  # 2 features
            [angle, eigenvalue_ratio, variance_explained]  # 3 features
        ])
        
        return features
    
    def fit(self, distribution_series):
        """
        Learn the normal pattern from a series of distributions.
        
        Parameters:
        distribution_series: list of arrays, each array of shape (n_points, 2)
        """
        # Extract features from each distribution
        features = []
        for dist in tqdm(distribution_series, desc="Extracting"):
            try:
                feat = self._extract_distribution_features(dist)
            except:
                continue
            if not np.isnan(feat).any():
                features.append(feat)
        
        features = np.array(features)
        
        # Fit isolation forest to identify normal pattern at distribution level
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.isolation_forest.fit(features)
        
        self.fitted = True
        return self
    
    def score_distribution(self, points):
        """
        Score a single distribution based on how well it matches the normal pattern.
        
        Parameters:
        points: array-like of shape (n_points, 2)
        
        Returns:
        dict with scores and classification
        """
        if not self.fitted:
            raise RuntimeError("Classifier must be fitted before scoring")
        
        # Extract features
        try:
            features = self._extract_distribution_features(points)
        except ValueError:
            return {
                'anomaly_score': np.nan,
            }
            
        # Get anomaly score from isolation forest
        anomaly_score = self.isolation_forest.score_samples([features])[0]

        return {
            'anomaly_score': anomaly_score,
        }
    
    def score_series(self, distribution_series):
        """
        Score a series of distributions.
        
        Parameters:
        distribution_series: list of arrays, each array of shape (n_points, 2)
        
        Returns:
        list of score dictionaries
        """
        scores = []
        for dist in tqdm(distribution_series, desc="Scoring"):
            scores.append(self.score_distribution(dist))
            
        return scores
    
    def plot_scores(self, scores):
        """
        Plot anomaly scores from a series of distributions.
        
        Parameters:
        scores: list of score dictionaries
        """
        anomaly_scores = [s['anomaly_score'] for s in scores]
        plt.hist(anomaly_scores, bins=20)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.show()