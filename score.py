
import time

from ifcb import DataDirectory

from dataloader import IFCB_ASPECT_RATIO

from classifier import load_extract_parallel, load_model, score_distributions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Score anomalies in point cloud data')
    parser.add_argument('data_dir', help='Directory containing point cloud data')
    parser.add_argument('--id-file', default=None, help='File containing list of IDs to load')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs for load/extraction phase')
    parser.add_argument('--aspect-ratio', type=float, default=IFCB_ASPECT_RATIO, help='Camera frame aspect ratio (width/height)')
    parser.add_argument('--chunk-size', type=int, default=100, help='Number of PIDs to process in each chunk')
    parser.add_argument('--model', default='classifier.pkl', help='Model load path')
    parser.add_argument('--output', default='scores.csv', help='Output CSV file path')

    args = parser.parse_args()
 
    beginning = time.time()
    
    print(f'Loading model from {args.model}')

    classifier = load_model(args.model)

    then = time.time()

    print(f'Extracting features from point clouds in {args.data_dir}')

    if args.id_file is not None:
        with open(args.id_file, 'r') as f:
            pids = [line.strip() for line in f]
    else:
        pids = []
        for bin in DataDirectory(args.data_dir):
            pids.append(bin.lid)

    feature_results = load_extract_parallel(pids, args.data_dir, aspect_ratio=args.aspect_ratio, n_jobs=args.n_jobs, chunk_size=args.chunk_size)

    elapsed = time.time() - then

    print(f'Extracted features for {len(feature_results)} point clouds in {elapsed:.2f} seconds')

    then = time.time()

    print(f'Scoring point clouds using classifier')

    results = score_distributions(classifier, feature_results)

    print(results)

    elapsed = time.time() - then

    print(f'Scored {len(results)} point clouds in {elapsed:.2f} seconds')

    print('Saving results ...')

    with open(args.output, 'w') as csv_file:
        csv_file.write('pid,anomaly_score\n')
        for scoredict in results:
            csv_file.write(f"{scoredict['pid']},{scoredict['anomaly_score']:.4f}\n")
    