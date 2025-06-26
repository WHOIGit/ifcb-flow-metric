# train.py
import argparse
import time
from ifcb import DataDirectory
from models.feature_extractor import FeatureExtractor
from models.trainer import ModelTrainer
from utils.constants import IFCB_ASPECT_RATIO, CONTAMINATION, CHUNK_SIZE, N_JOBS, MODEL

def main():
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    parser.add_argument('data_dir', help='Directory containing point cloud data')
    parser.add_argument('--id-file', help='File containing list of IDs to load')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS, help='Number of parallel jobs for load/extraction phase')
    parser.add_argument('--contamination', type=float, default=CONTAMINATION, help='Expected fraction of anomalous distributions')
    parser.add_argument('--aspect-ratio', type=float, default=IFCB_ASPECT_RATIO, help='Camera frame aspect ratio (width/height)')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Number of PIDs to process in each chunk')
    parser.add_argument('--model', default=MODEL, help='Model save/load path')
    
    args = parser.parse_args()

    beginning = time.time()

    # Load PIDs
    if args.id_file:
        with open(args.id_file) as f:
            pids = [line.strip() for line in f]
    else:
        pids = [bin.lid for bin in DataDirectory(args.data_dir)]

    then = time.time()
    
    print(f'Loading and performing feature extraction on {len(pids)} point clouds')
    
    # Extract features from point clouds
    extractor = FeatureExtractor(aspect_ratio=args.aspect_ratio)
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

    elapsed = time.time() - then

    print(f'Trained classifier in {elapsed:.2f} seconds')

    # save the classifier
    trainer.save_model(classifier)

    elapsed = time.time() - beginning

    print(f'Total load/extract/train time: {elapsed:.2f} seconds')

if __name__ == "__main__":
    main()
