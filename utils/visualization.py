import matplotlib.pyplot as plt

def plot_scores(scores):
    """
    Plot anomaly scores from a series of distributions.
    Utility function for visualization during development/debugging.
    
    Parameters:
    scores: list of score dictionaries
    """
    anomaly_scores = [s['anomaly_score'] for s in scores]
    plt.hist(anomaly_scores, bins=20)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.show()

