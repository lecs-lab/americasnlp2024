import pandas as pd
import argparse


def copy_target_column(source_path, destination_path):
    # Load the source and destination files
    source_df = pd.read_csv(source_path, sep='\t', header=None)
    destination_df = pd.read_csv(destination_path, sep='\t')

    source_df.columns = ['Target']
    destination_df.columns = ['ID', 'Source', 'Change', 'Target']

    if not destination_df.isnull().iloc[1]['Target']:
        destination_df['Predicted Target'] = source_df['Target']
    else:
        destination_df['Target'] = source_df['Target']

    # Save the updated destination DataFrame to a new file
    destination_df.to_csv(source_path, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Copy "Target" column from a prediction TSV file to the full file.')
    parser.add_argument('source_path', type=str,
                        help='Path to the source TSV file.')
    parser.add_argument('destination_path', type=str,
                        help='Path to the destination TSV file.')

    args = parser.parse_args()

    copy_target_column(args.source_path, args.destination_path)


if __name__ == "__main__":
    main()
