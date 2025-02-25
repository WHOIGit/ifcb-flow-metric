import numpy as np

class Inferencer:
    """Performs inference using trained model."""
    
    def __init__(self, model):
        self.model = model

    def score_distributions(self, feature_results):
        """
        Score a series of distributions using a trained classifier.
        
        Parameters:
        classifier: trained IsolationForest instance
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
        } for pid, score in zip(pids, self.model.score_samples(features))]

        anomaly_scores.extend([{'pid': pid, 'anomaly_score': np.nan} for pid in bad_pids])
        
        return anomaly_scores

