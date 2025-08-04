# score.py
import argparse
import csv
import time
from ifcb import DataDirectory
from models.feature_extractor import FeatureExtractor
from models.trainer import ModelTrainer
from models.inference import Inferencer
from utils.constants import IFCB_ASPECT_RATIO, CHUNK_SIZE, N_JOBS, MODEL, SCORES_OUTPUT
from utils.feature_config import load_feature_config

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


    trainer = ModelTrainer(filepath=args.model, n_jobs=args.n_jobs)
    
    classifier = trainer.load_model()

    then = time.time()

    print(f'Extracting features from point clouds in {args.data_dir}')
    # Load PIDs and model
    if args.id_file:
        with open(args.id_file) as f:
            pids = [line.strip() for line in f]
    else:
        pids = [bin.lid for bin in DataDirectory(args.data_dir, require_roi_files=False)]
    
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
    
    # Extract features
    feature_results = extractor.load_extract_parallel(
        pids, args.data_dir,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size
    )

    elapsed = time.time() - then

    print(f'Extracted features for {len(feature_results)} point clouds in {elapsed:.2f} seconds')

    then = time.time()

    print(f'Scoring point clouds using classifier')

    # Score distributions
    inferencer = Inferencer(classifier)
    scores = inferencer.score_distributions(feature_results)

    elapsed = time.time() - then

    print(f'Scored {len(scores)} point clouds in {elapsed:.2f} seconds')

    print('Saving results ...')
    # Save results
    with open(args.output, 'w') as csv_file:
        csv_file.write('pid,anomaly_score\n')
        for scoredict in scores:
            csv_file.write(f"{scoredict['pid']},{scoredict['anomaly_score']:.4f}\n")

if __name__ == '__main__':
    main()