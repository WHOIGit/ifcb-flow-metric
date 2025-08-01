# create_violin_plot.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description='Evaluate IFCB flow metric model by comparing score distributions')
    parser.add_argument('csv1_path', help='Path to first CSV file with anomaly scores')
    parser.add_argument('csv2_path', help='Path to second CSV file with anomaly scores')
    parser.add_argument('--output', required=True, help='Output path for the violin plot image')
    parser.add_argument('--title', default='Score Distributions by Category (Violin Plot)', 
                       help='Title for the violin plot')
    parser.add_argument('--name1', default='Dataset 1', help='Name for first distribution')
    parser.add_argument('--name2', default='Dataset 2', help='Name for second distribution')
    
    args = parser.parse_args()
    
    # Load the CSV files
    print(f'Loading {args.csv1_path}')
    df1 = pd.read_csv(args.csv1_path)
    df1['category'] = args.name1
    
    print(f'Loading {args.csv2_path}')
    df2 = pd.read_csv(args.csv2_path)
    df2['category'] = args.name2
    
    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    print(f'Dataset 1 ({args.name1}): {len(df1)} samples')
    print(f'Dataset 2 ({args.name2}): {len(df2)} samples')
    
    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=combined_df, x='category', y='anomaly_score')
    plt.title(args.title)
    plt.xlabel('Category')
    plt.ylabel('Anomaly Score')
    
    # Add some statistics to the plot
    for i, category in enumerate([args.name1, args.name2]):
        data = combined_df[combined_df['category'] == category]['anomaly_score']
        median_val = data.median()
        mean_val = data.mean()
        plt.text(i, median_val, f'Median: {median_val:.3f}', 
                horizontalalignment='center', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    print(f'Saving violin plot to {args.output}')
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print('\nSummary Statistics:')
    print(combined_df.groupby('category')['anomaly_score'].describe())


if __name__ == '__main__':
    main()
