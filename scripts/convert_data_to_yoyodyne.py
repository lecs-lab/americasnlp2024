import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Convert data to Yoyodyne format')
parser.add_argument('input_file', help='Path to the input TSV file')
parser.add_argument('output_file', help='Path to the output TSV file')

args = parser.parse_args()

# Read the input file using pandas
data = pd.read_csv(args.input_file, delimiter='\t')

# Remove the ID column
data = data.drop(['ID', 'Target'], axis=1)

# Write the modified data to the output file
data.to_csv(args.output_file, sep='\t', index=False, header=False)
