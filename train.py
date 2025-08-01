# train.py
import argparse
import time
from ifcb import DataDirectory
from models.feature_extractor import FeatureExtractor
from models.trainer import ModelTrainer
from utils.constants import IFCB_ASPECT_RATIO, CONTAMINATION, CHUNK_SIZE, N_JOBS, MODEL
from utils.feature_config import load_feature_config

def main():
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    parser.add_argument('data_dir', help='Directory containing point cloud data')
    parser.add_argument('--id-file', help='File containing list of IDs to load')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS, help='Number of parallel jobs for load/extraction phase')
    parser.add_argument('--contamination', type=float, default=CONTAMINATION, help='Expected fraction of anomalous distributions')
    parser.add_argument('--aspect-ratio', type=float, default=IFCB_ASPECT_RATIO, help='Camera frame aspect ratio (width/height)')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Number of PIDs to process in each chunk')
    parser.add_argument('--model', default=MODEL, help='Model save/load path')
    parser.add_argument('--feature-config', help='YAML file specifying which features to use for training')
    
    args = parser.parse_args()

    beginning = time.time()

    # Load PIDs
    if args.id_file:
        with open(args.id_file) as f:
            pids = [line.strip() for line in f]
    else:
        pids = [bin.lid for bin in DataDirectory(args.data_dir, require_roi_files=False)]

    then = time.time()
    
    print(f'Loading and performing feature extraction on {len(pids)} point clouds')
    
    # Load feature configuration if provided
    feature_config = None
    if args.feature_config:
        print(f'Loading feature configuration from {args.feature_config}')
        feature_config = load_feature_config(args.feature_config)
        
        # Count enabled features for reporting
        enabled_count = sum(1 for category in feature_config.values() 
                          if isinstance(category, dict)
                          for enabled in category.values() if enabled)
        print(f'Using {enabled_count} enabled features')
    else:
        print('Using default feature configuration (all features enabled)')
    
    # Extract features from point clouds
    extractor = FeatureExtractor(aspect_ratio=args.aspect_ratio, feature_config=feature_config)
    feature_results = extractor.load_extract_parallel(
        pids, args.data_dir,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size
    )

    elapsed = time.time() - then

    print(f'Extracted features for {len(feature_results)} point clouds in {elapsed:.2f} seconds')

    then = time.time()

    # Train the classifier
    print(f'Training classifier')
    trainer = ModelTrainer(filepath=args.model, contamination=args.contamination, n_jobs=args.n_jobs)
    
    classifier = trainer.train_classifier(feature_results)

    print(f'Trained classifier in {elapsed:.2f} seconds')

    # save the classifier
    trainer.save_model(classifier)

    elapsed = time.time() - beginning

    print(f'Total load/extract/train time: {elapsed:.2f} seconds')

if __name__ == "__main__":
    main()
