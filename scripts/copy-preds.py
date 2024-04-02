import pandas as pd
import argparse


def copy_target_column(source_path, destination_path):
    # Load the source and destination files
    source_df = pd.read_csv(source_path, sep='\t')
    destination_df = pd.read_csv(destination_path, sep='\t')

    # Check if the 'Target' column exists in the source
    if 'Target' not in source_df.columns:
        raise ValueError("'Target' column not found in the source file.")

    # Replace or add the 'Target' column in the destination DataFrame
    destination_df['Predicted Target'] = source_df['Target']

    # Save the updated destination DataFrame to a new file
    destination_df.to_csv(source_path, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Copy "Target" column from a prediction TSV file to the full file.')
    parser.add_argument('source_path', type=str, help='Path to the source TSV file.')
    parser.add_argument('destination_path', type=str, help='Path to the destination TSV file.')

    args = parser.parse_args()

    copy_target_column(args.source_path, args.destination_path)


if __name__ == "__main__":
    main()
