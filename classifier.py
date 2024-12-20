import pickle
import time
from typing import Any, Dict, List
from joblib import Parallel, delayed
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from tqdm import tqdm

from ifcb import DataDirectory

from dataloader import IFCB_ASPECT_RATIO, get_points
from utilities import parallel_map

    
def plot_scores(scores):
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


def extract_features(load_result, aspect_ratio = IFCB_ASPECT_RATIO):
    """Extract statistical features from a single point cloud distribution."""

    try:
        pid = load_result['pid']
        points = load_result['points']

        # some features won't converge or be useful if there are too few points
        if points.shape[0] < 30:  # 30 because 20 is the minimum for LOF
            raise ValueError("Distribution has too few points")

        # Normalize points to account for aspect ratio
        normalized_points = points.copy()
        normalized_points[:, 0] = normalized_points[:, 0] / aspect_ratio

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
        
        # delete everything but the features to save memory
        del normalized_points
        del gmm
        del lof
        del pca

        return { 'pid': pid, 'features': features }
    
    except Exception as e:

        return { 'pid': pid, 'features': None }


def load_extract(pids, directory, aspect_ratio = IFCB_ASPECT_RATIO):
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
    feature_results = [extract_features(load_result, aspect_ratio) for load_result in load_results]

    del load_results # discard point clouds to save memory

    return feature_results


def load_extract_parallel(
    pids: List[str],
    directory: str,
    chunk_size: int = 100,
    n_jobs: int = -1,
    aspect_ratio: float = IFCB_ASPECT_RATIO
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
        delayed(load_extract)(chunk, directory, aspect_ratio)
        for chunk in tqdm(chunks, desc="Processing chunks")
    )
    
    # Flatten results
    flattened_results = []
    for chunk_result in results:
        flattened_results.extend(chunk_result)
    
    print(f"Processed {len(flattened_results)} PIDs successfully")
    
    return flattened_results


def train_classifier(feature_results, contamination=0.1, n_jobs: int = -1):
    """
    Train a classifier using a list of feature results.
    
    Parameters:
    feature_results: list of feature dictionaries
    contamination: float, expected fraction of anomalous distributions
    """
    features = []
    for result in feature_results:
        if result['features'] is not None:
            features.append(result['features'])
    
    features = np.array(features)
    
    # Fit isolation forest to identify normal pattern at distribution level
    isolation_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=n_jobs
    )
    isolation_forest.fit(features)
    
    return isolation_forest


def score_distributions(classifier, feature_results):
    """
    Score a series of distributions using a trained classifier.
    
    Parameters:
    classifier: trained IsolationForest instance
    load_results: list of load results
    """
    features = []
    pids = []
    bad_pids = []
    for result in feature_results:
        if result['features'] is not None:
            pids.append(result['pid'])
            features.append(result['features'])
        else:
            bad_pids.append(result['pid'])
    
    features = np.array(features)
    
    # Get anomaly scores from isolation forest
    anomaly_scores = [{
        'pid': pid,
        'anomaly_score': score
    } for pid, score in zip(pids, classifier.score_samples(features))]

    anomaly_scores.extend([{'pid': pid, 'anomaly_score': np.nan} for pid in bad_pids])
    
    return anomaly_scores


def save_model(classifier, filepath: str):
    """
    Save the classifier to a file.
    
    Parameters:
    classifier: trained IsolationForest instance
    filepath: str, path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(classifier, f)


def load_model(filepath: str):
    """
    Load a classifier from a file.
    
    Parameters:
    filepath: str, path to load the model from
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)