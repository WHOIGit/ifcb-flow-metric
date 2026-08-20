# score.py
import argparse
import csv
import time

from ifcb_flow_metric.models.feature_extractor import FeatureExtractor
from ifcb_flow_metric.models.trainer import ModelTrainer
from ifcb_flow_metric.models.inference import Inferencer
from ifcb_flow_metric.utils.constants import IFCB_ASPECT_RATIO, CHUNK_SIZE, N_JOBS, MODEL, SCORES_OUTPUT
from ifcb_flow_metric.utils.dataloader import get_pid_pairs, summarize_failures
from ifcb_flow_metric.utils.feature_config import load_feature_config

def main():
    parser = argparse.ArgumentParser(description='Score anomalies in point cloud data')
    parser.add_argument('data_dir', help='Directory containing point cloud data')
    parser.add_argument('--id-file', default=None, help='File containing list of IDs to load')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS, help='Number of parallel jobs for load/extraction phase')
    parser.add_argument('--aspect-ratio', type=float, default=IFCB_ASPECT_RATIO)
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Number of PIDs to process in each chunk')
    parser.add_argument('--model', default=MODEL, help='Model load path')
    parser.add_argument('--output', default=SCORES_OUTPUT, help='Output CSV file path')
    # Feature configuration options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('--config', help='YAML string specifying which features to use for inference')
    config_group.add_argument('--config-file', help='YAML file path specifying which features to use for inference')
    args = parser.parse_args()

    beginning = time.time()

    print(f'Loading model from {args.model}')

    # Load model first to get feature configuration
    trainer = ModelTrainer(filepath=args.model, n_jobs=args.n_jobs)
    classifier = trainer.load_model()

    # Check if model has stored feature names
    if hasattr(classifier, 'feature_names_'):
        model_feature_names = classifier.feature_names_
        print(f'Model was trained with {len(model_feature_names)} features: {model_feature_names}')

        # Create feature config from model's feature names
        feature_config = {}
        all_feature_categories = {
            'spatial_stats': ['mean_x', 'mean_y', 'std_x', 'std_y', 'median_x', 'median_y', 'iqr_x', 'iqr_y'],
            'distribution_shape': ['ratio_spread', 'core_fraction'],
            'clipping_detection': ['duplicate_fraction', 'max_duplicate_fraction'],
            'histogram_uniformity': ['cv_x', 'cv_y'],
            'statistical_moments': ['skew_x', 'skew_y', 'kurt_x', 'kurt_y'],
            'pca_orientation': ['angle', 'eigen_ratio'],
            'edge_features': ['left_edge_fraction', 'right_edge_fraction', 'top_edge_fraction', 'bottom_edge_fraction', 'total_edge_fraction'],
            'temporal': ['t_y_var'],
            'trigger_stats': ['roi_trigger_fraction']
        }

        # Build feature config based on what was used in training
        for category, features in all_feature_categories.items():
            feature_config[category] = {}
            for feature in features:
                feature_config[category][feature] = feature in model_feature_names

        print('Using feature configuration from trained model')
    else:
        print('Warning: Model does not have stored feature names, using provided configuration or defaults')

        # Load feature configuration if provided
        feature_config = None
        if args.config:
            print('Loading feature configuration from YAML string')
            import yaml
            feature_config = yaml.safe_load(args.config)

            # Count enabled features for reporting
            enabled_count = sum(1 for category in feature_config.values()
                              if isinstance(category, dict)
                              for enabled in category.values() if enabled)
            print(f'Using {enabled_count} enabled features')
        elif args.config_file:
            print(f'Loading feature configuration from {args.config_file}')
            feature_config = load_feature_config(args.config_file)

            # Count enabled features for reporting
            enabled_count = sum(1 for category in feature_config.values()
                              if isinstance(category, dict)
                              for enabled in category.values() if enabled)
            print(f'Using {enabled_count} enabled features')
        else:
            print('Using default feature configuration (all features enabled)')

    extractor = FeatureExtractor(aspect_ratio=args.aspect_ratio, feature_config=feature_config)

    # Single pass over the data tree: pid -> adc path mapping,
    # restricted to the ID file if given
    then = time.time()
    print(f'Listing filesets in {args.data_dir}')
    pairs = get_pid_pairs(args.data_dir, args.id_file)
    pids = [pid for pid, _ in pairs]
    print(f'Found {len(pids)} PIDs in {time.time() - then:.2f} seconds')

    # Extract features
    then = time.time()
    print(f'Extracting features from point clouds in {args.data_dir}')
    feature_results = extractor.load_extract_parallel(
        pairs,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size
    )

    elapsed = time.time() - then

    print(f'Extracted features for {len(feature_results)} point clouds in {elapsed:.2f} seconds')

    summarize_failures(feature_results)

    then = time.time()

    print('Scoring point clouds using classifier')

    # Score distributions
    inferencer = Inferencer(classifier)
    scores = inferencer.score_distributions(feature_results)

    elapsed = time.time() - then

    print(f'Scored {len(scores)} point clouds in {elapsed:.2f} seconds')

    print('Saving results ...')
    # Save results; PIDs that failed to produce features carry their error
    # in the third column (anomaly_score is 'nan' for those)
    errors = {r['pid']: r.get('error') for r in feature_results}
    with open(args.output, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['pid', 'anomaly_score', 'error'])
        for scoredict in scores:
            writer.writerow([
                scoredict['pid'],
                f"{scoredict['anomaly_score']:.4f}",
                errors.get(scoredict['pid']) or '',
            ])

if __name__ == '__main__':
    main()
