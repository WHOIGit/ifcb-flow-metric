import numpy as np
import sys
import pickle
from typing import List, Dict, Any, Iterable
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import AdcLoader
from classifier import DistributionSeriesClassifier

class PointCloudClassifier:
    def __init__(self, loader, contamination: float = 0.1):
        """
        Initialize classifier with function to retrieve point clouds.
        
        Parameters:
        get_points_fn: Callable that takes an ID and returns numpy array of points
        contamination: Expected fraction of anomalous distributions
        """
        self.loader = loader
        self.classifier = DistributionSeriesClassifier(contamination=contamination)
        self.is_trained = False
    
    def get_points(self, cloud_id: str) -> np.ndarray:
        """Get points for a single point cloud by ID."""
        return self.loader[cloud_id]
    
    def train(self, training_ids: List[str]) -> Dict[str, Any]:
        """
        Train classifier using list of point cloud IDs.
        
        Parameters:
        training_ids: List of IDs to use for training
        
        Returns:
        Dictionary with training results and statistics
        """
        print(f"Training on {len(training_ids)} distributions...")
        
        # Load all training distributions
        distributions = []
        for cloud_id in tqdm(training_ids):
            points = self.get_points(cloud_id)
            distributions.append(points)
        
        # Train the classifier
        self.classifier.fit(distributions)
        self.is_trained = True
        
        # Get training set scores
        scores = self.classifier.score_series(distributions)
        
        # Compute training statistics
        training_stats = {
        }

        return training_stats
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    @classmethod
    def load_model(cls, filepath: str, loader):
        """Load trained model from file."""
        instance = cls(loader)
        
        with open(filepath, 'rb') as f:
            instance.classifier = pickle.load(f)
            instance.is_trained = True
        
        return instance
    
    def classify_point_cloud(self, cloud_id: str, visualize: bool = False) -> Dict[str, Any]:
        """
        Classify a single point cloud by ID.
        
        Parameters:
        cloud_id: ID of point cloud to classify
        visualize: Whether to generate visualization
        
        Returns:
        Dictionary with classification results
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference")
        
        # Get points for this cloud
        points = self.get_points(cloud_id)
        
        # Get scores
        scores = self.classifier.score_distribution(points)
        
        # Create visualization if requested
        fig = None
        if visualize:
            fig, ax = plt.subplots(figsize=(8, 8))
            self.classifier.visualize_distribution(points, scores, ax=ax)
        
        return {
            'cloud_id': cloud_id,
            'scores': scores,
            'figure': fig
        }
    
    def classify_many(self, cloud_ids: Iterable[str], 
                     visualize: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify multiple point clouds by ID.
        
        Parameters:
        cloud_ids: List of point cloud IDs to classify
        visualize: Whether to generate visualizations
        
        Returns:
        Dictionary with results and summary statistics
        """
        results = {}
        for cloud_id in cloud_ids:
            try:
                result = self.classify_point_cloud(cloud_id, visualize)
                results[cloud_id] = result
            except:
                pass
    
        return results

# Example usage:
if __name__ == "__main__":
    # Example data loader
    loader = AdcLoader(sys.argv[1])
    
    # Create classifier with data access function
    classifier = PointCloudClassifier(loader)

    # attempt to train on all data, some may be bad
    training_ids = list(loader)

    # Train the classifier
    training_stats = classifier.train(training_ids)
    print("\nTraining complete!")
    print("Training statistics:")
    for key, value in training_stats.items():
        print(f"{key}: {value}")
    
    # Save the model
    classifier.save_model('cloud_classifier.pkl')
    
    # Load model (would normally be in a different session)
    classifier = PointCloudClassifier.load_model('cloud_classifier.pkl', loader)
    
    # Inference example
    results = classifier.classify_many(training_ids)

    csv_file = open('cloud_classification.csv', 'w')
    for key, value in results.items():
        csv_file.write(f"{key},{value['scores']['anomaly_score']}\n")
    
