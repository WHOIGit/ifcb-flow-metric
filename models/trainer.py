from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import pickle
from utils.constants import CONTAMINATION, N_JOBS, RANDOM_STATE

class ModelTrainer:
    """Trains a model on extracted features."""
    
    def __init__(self, filepath: str, contamination=CONTAMINATION, n_jobs=N_JOBS, max_samples='auto', max_features=1.0):
        self.filepath = filepath
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.max_samples = max_samples
        self.max_features = max_features

    def train_classifier(self, feature_df: pd.DataFrame):
        """
        Train a classifier using a DataFrame of features.
        
        Parameters:
        feature_df: pandas DataFrame with 'pid' column and feature columns
        """
        # Separate PIDs from features
        feature_columns = [col for col in feature_df.columns if col != 'pid']
        features = feature_df[feature_columns].values
        
        print(f"Training on {len(feature_df)} samples with {len(feature_columns)} features")
        print(f"Feature columns: {feature_columns}")
        
        # Fit isolation forest to identify normal pattern at distribution level
        isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=RANDOM_STATE,
            n_jobs=self.n_jobs,
            max_samples=self.max_samples,
            max_features=self.max_features
        )
        isolation_forest.fit(features)
        
        # Store feature names in the model for later reference
        isolation_forest.feature_names_ = feature_columns
        
        return isolation_forest

    def save_model(self, classifier):
        """
        Save the classifier to a file.
        
        Parameters:
        classifier: trained IsolationForest instance
        filepath: str, path to save the model
        """
        with open(self.filepath, 'wb') as f:
            pickle.dump(classifier, f)


    def load_model(self):
        """
        Load a classifier from a file.
        
        Parameters:
        filepath: str, path to load the model from
        """
        with open(self.filepath, 'rb') as f:
            classifier = pickle.load(f)
            return classifier