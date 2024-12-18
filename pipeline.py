import numpy as np
import sys
import pickle
from typing import List, Dict, Any, Iterable
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import AdcLoader
from classifier import DistributionSeriesClassifier

class PointCloudClassifier:
    def __init__(self, loader, aspect_ratio: float = 1.0, contamination: float = 0.1):
        """
        Initialize classifier with function to retrieve point clouds.
        
        Parameters:
        loader: AdcLoader instance to load point cloud data
        aspect_ratio: float, width/height ratio of the camera frame
        contamination: Expected fraction of anomalous distributions
        """
        self.loader = loader
        self.classifier = DistributionSeriesClassifier(
            aspect_ratio=aspect_ratio,
            contamination=contamination
        )
        self.is_trained = False
        self.aspect_ratio = aspect_ratio
    
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
        for cloud_id in tqdm(training_ids, desc="Loading"):
            try:
                points = self.get_points(cloud_id)
                distributions.append(points)
            except Exception as e:
                print(f"Warning: Could not load points for {cloud_id}: {str(e)}")
                continue
        
        # Train the classifier
        self.classifier.fit(distributions)
        self.is_trained = True
        
        # Get training set scores
        scores = self.classifier.score_series(distributions)
        
        # Compute basic training statistics
        anomaly_scores = [s['anomaly_score'] for s in scores if not np.isnan(s['anomaly_score'])]
        training_stats = {
            'n_distributions': len(distributions),
            'n_scored': len(anomaly_scores),
            'mean_score': np.mean(anomaly_scores),
            'std_score': np.std(anomaly_scores)
        }

        return training_stats
    
    def save_model(self, filepath: str):
        """Save trained model and parameters to file."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        save_dict = {
            'classifier': self.classifier,
            'aspect_ratio': self.aspect_ratio
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load_model(cls, filepath: str, loader):
        """Load trained model and parameters from file."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        aspect_ratio = save_dict['aspect_ratio']
        instance = cls(loader, aspect_ratio=aspect_ratio)
        instance.classifier = save_dict['classifier']
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
        try:
            points = self.get_points(cloud_id)
        except Exception as e:
            print(f"Warning: Could not load points for {cloud_id}: {str(e)}")
            return {
                'cloud_id': cloud_id,
                'scores': {'anomaly_score': np.nan},
                'error': str(e)
            }
        
        # Get scores
        scores = self.classifier.score_distribution(points)
        
        # Create visualization if requested
        fig = None
        if visualize:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(points[:, 0], points[:, 1], alpha=0.5, s=1)
            ax.set_title(f'Cloud {cloud_id} (Anomaly Score: {scores["anomaly_score"]:.3f})')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_aspect(1/self.aspect_ratio)  # Correct aspect ratio for display
        
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
        for cloud_id in tqdm(cloud_ids, desc="Classifying"):
            result = self.classify_point_cloud(cloud_id, visualize)
            results[cloud_id] = result
    
        return results

# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify point cloud distributions')
    parser.add_argument('data_dir', help='Directory containing point cloud data')
    parser.add_argument('--aspect-ratio', type=float, default=1.0,
                       help='Camera frame aspect ratio (width/height)')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Expected fraction of anomalous distributions')
    parser.add_argument('--output', default='cloud_classification.csv',
                       help='Output CSV file path')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--model', default=None,
                       help='Model save/load path')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for each cloud')
    
    args = parser.parse_args()
    
    # Initialize loader and classifier
    loader = AdcLoader(args.data_dir)
    classifier = PointCloudClassifier(
        loader,
        aspect_ratio=args.aspect_ratio,
        contamination=args.contamination
    )
    
    # Get all available training IDs
    training_ids = list(loader)
    
    # Train or load the classifier
    if not args.train:
        print(f"Loading model from {args.model}")
        classifier = PointCloudClassifier.load_model(args.model, loader)
    else:
        print(f"Training new model")
        training_stats = classifier.train(training_ids)
        print("Training stats:", training_stats)
        classifier.save_model(args.model)
    
    # Run inference
    results = classifier.classify_many(training_ids, visualize=args.visualize)
    
    # Save results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as csv_file:
        csv_file.write("cloud_id,anomaly_score\n")
        for cloud_id, result in results.items():
            score = result['scores']['anomaly_score']
            csv_file.write(f"{cloud_id},{score}\n")