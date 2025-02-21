from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from utils.constants import IFCB_ASPECT_RATIO, EDGE_TOLERANCE
from utils.dataloader import get_points

class FeatureExtractor:
    """Extracts features from point cloud distributions."""
    
    def __init__(self, aspect_ratio=IFCB_ASPECT_RATIO, edge_tolerance=EDGE_TOLERANCE):
        """ 
        Parameters:
        points: nx2 array of x,y coordinates
        edge_tolerance: distance from edge to consider a point "on edge
        """
        self.aspect_ratio = aspect_ratio
        self.edge_tolerance = edge_tolerance

    def extract_edge_features(self, points):
        """Extract features related to points on frame edges.
        
        Parameters:
        points: nx2 array of x,y coordinates

        Returns:
        Array of edge-related features
        """
        # Get frame dimensions
        x_max, y_max = points.max(axis=0)
        
        # Find points near edges
        left_edge = points[points[:,0] <= self.edge_tolerance]
        right_edge = points[points[:,0] >= x_max - self.edge_tolerance]
        top_edge = points[points[:,1] <= self.edge_tolerance]
        bottom_edge = points[points[:,1] >= y_max - self.edge_tolerance]
        
         # Calculate features
        edge_fractions = [
            len(edge) / len(points) for edge in 
            [left_edge, right_edge, top_edge, bottom_edge]
        ]
        
        total_edge_fraction = sum(len(edge) for edge in [left_edge, right_edge, top_edge, bottom_edge]) / len(points)
        
        return np.array(edge_fractions + [total_edge_fraction])

    def extract_features(self, load_result):
        """Extract statistical features from a single point cloud distribution."""

        try:
            pid = load_result['pid']
            points = load_result['points']

            # some features won't converge or be useful if there are too few points
            if points.shape[0] < 30:  # 30 because 20 is the minimum for LOF
                raise ValueError("Distribution has too few points")

            # Normalize points to account for aspect ratio
            normalized_points = points.copy()
            normalized_points[:, 0] = normalized_points[:, 0] / self.aspect_ratio

            # Edge features (on original points)
            edge_features = self.extract_edge_features(points)

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
                [angle, eigenvalue_ratio, variance_explained],  # 3 features
                edge_features
            ])
            
            # delete everything but the features to save memory
            del normalized_points
            del gmm
            del lof
            del pca

            return { 'pid': pid, 'features': features }
        
        except Exception as e:

            return { 'pid': pid, 'features': None }


    def load_extract(self, pids, directory):
        """
        Load and extract features from a list of point cloud distributions.
        
        Parameters:

        pids: list of distribution IDs
        directory: str, path to the data directory
        aspect_ratio: float, width/height ratio of the camera frame

        Returns:

        List of feature dictionaries
        """

        load_results = [get_points(pid, directory) for pid in pids]
        feature_results = [self.extract_features(load_result) for load_result in load_results]

        del load_results # discard point clouds to save memory

        return feature_results


    def load_extract_parallel(
        self,
        pids: List[str],
        directory: str,
        chunk_size: int = 100,
        n_jobs: int = -1,
    ) -> List[Dict[str, Any]]:
        """
        Process PIDs in parallel chunks to manage memory usage
        
        Parameters:
        pids: List of all PIDs to process
        directory: Data directory path
        chunk_size: Number of PIDs to process in each chunk
        n_jobs: Number of parallel jobs (-1 for all cores)
        aspect_ratio: Camera frame aspect ratio
        
        Returns:
        List of feature dictionaries
        """
        # Create chunks
        chunks = [pids[i:i + chunk_size] for i in range(0, len(pids), chunk_size)]
        
        print(f"Processing {len(pids)} PIDs in {len(chunks)} chunks of size {chunk_size}")
        
        # Process chunks in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.load_extract)(chunk, directory)
            for chunk in tqdm(chunks, desc="Processing chunks")
        )
        
        # Flatten results
        flattened_results = []
        for chunk_result in results:
            flattened_results.extend(chunk_result)
        
        print(f"Processed {len(flattened_results)} PIDs successfully")
        
        return flattened_results