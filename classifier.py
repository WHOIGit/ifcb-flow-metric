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
    the dominant pattern in the dataset.
    """
    
    def __init__(self, contamination=0.1, n_components=2):
        """
        Initialize classifier.
        
        Parameters:
        contamination: float, expected fraction of anomalous distributions
        n_components: int, number of Gaussian components for modeling core pattern
        """
        self.contamination = contamination
        self.n_components = n_components
        self.fitted = False
    
    def _extract_distribution_features(self, points):
        """Extract statistical features from a single point cloud distribution."""
        # Some features won't converge or be useful if there are too few points, simply exclude those distributions
        if points.shape[0] < 30: # 30 because 20 is the minimum for LOF
            raise ValueError("Distribution has too few points")

        # Existing GMM features
        gmm = GaussianMixture(n_components=self.n_components, random_state=42)
        gmm.fit(points)
        
        means = gmm.means_
        covs = gmm.covariances_
        weights = gmm.weights_
        
        # Sort by weight for consistent ordering
        sort_idx = np.argsort(weights)[::-1]
        means = means[sort_idx]
        covs = covs[sort_idx]
        weights = weights[sort_idx]
        
        # Basic stats
        center = np.mean(points, axis=0)
        spread = np.std(points, axis=0)
        
        # LOF features
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(points)
        lof_scores = -lof.negative_outlier_factor_
        
        # PCA features
        pca = PCA(n_components=2)
        pca.fit(points)
        
        # First component angle (in radians)
        first_component = pca.components_[0]
        angle = np.arctan2(first_component[1], first_component[0])
        
        # Ratio of eigenvalues (indicates shape elongation)
        eigenvalue_ratio = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]
        
        # Percent variance explained by first component
        variance_explained = pca.explained_variance_ratio_[0]
        
        # Combine all features
        features = np.concatenate([
            means.flatten(),
            covs.flatten(),
            weights,
            center,
            spread,
            [np.mean(lof_scores), np.std(lof_scores)],
            [angle, eigenvalue_ratio, variance_explained]
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
        
        # Fit isolation forest to identify normal pattern at distribution level
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
        )
        self.isolation_forest.fit(features)
        
        # Store reference pattern
        # scores = self.isolation_forest.score_samples(features)
        # good_idx = np.array(scores >= np.percentile(scores, self.contamination * 100), dtype=bool)
        # self.reference_features = features[good_idx]
        
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
