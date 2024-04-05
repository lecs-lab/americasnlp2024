import argparse
import pandas as pd


def conjoin_dataframes(file1, file2, output_file):
    # Read the TSV files into pandas dataframes
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')

    # Concatenate the dataframes
    concatenated_df = pd.concat([df1, df2])

    # Shuffle the rows
    shuffled_df = concatenated_df.sample(frac=1).reset_index(drop=True)

    # Write the shuffled dataframe to the output file
    shuffled_df.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', help='Path to the first TSV file')
    parser.add_argument('file2', help='Path to the second TSV file')
    parser.add_argument('output_file', help='Path to the output file')
    args = parser.parse_args()

    # Call the conjoin_dataframes function with the input arguments
    conjoin_dataframes(args.file1, args.file2, args.output_file)
