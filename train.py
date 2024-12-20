import time 

from ifcb import DataDirectory

from classifier import load_extract_parallel, save_model, train_classifier
from dataloader import IFCB_ASPECT_RATIO


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a classifier on point cloud data')
    parser.add_argument('data_dir', help='Directory containing point cloud data')
    parser.add_argument('--id-file', default=None, help='File containing list of IDs to load')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--contamination', type=float, default=0.1, help='Expected fraction of anomalous distributions')
    parser.add_argument('--aspect-ratio', type=float, default=IFCB_ASPECT_RATIO, help='Camera frame aspect ratio (width/height)')
    parser.add_argument('--chunk-size', type=int, default=100, help='Number of PIDs to process in each chunk')
    parser.add_argument('--model', default='classifier.pkl', help='Model save/load path')
    
    args = parser.parse_args()
    
    beginning = time.time()

    if args.id_file is not None:
        with open(args.id_file, 'r') as f:
            pids = [line.strip() for line in f]
    else:
        pids = []
        for bin in DataDirectory(args.data_dir):
            pids.append(bin.lid)

    then = time.time()
    
    print(f'Loading and performing feature extraction on {len(pids)} point clouds')
    
    # Extract features from point clouds
    feature_results = load_extract_parallel(pids, args.data_dir, aspect_ratio=args.aspect_ratio, n_jobs=args.n_jobs, chunk_size=args.chunk_size)
    
    elapsed = time.time() - then

    print(f'Extracted features for {len(feature_results)} point clouds in {elapsed:.2f} seconds')

    then = time.time()

    # Train the classifier

    print(f'Training classifier')

    classifier = train_classifier(feature_results, contamination=args.contamination, n_jobs=args.n_jobs)

    elapsed = time.time() - then

    print(f'Trained classifier in {elapsed:.2f} seconds')

    # save the classifier
    save_model(classifier, args.model)

    elapsed = time.time() - beginning

    print(f'Total load/extract/train time: {elapsed:.2f} seconds')

