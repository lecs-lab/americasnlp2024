import pandas as pd

for lang in ['maya']:
    for num in [2, 3, 4]:

        # Specify the file path
        file_path = f'./final-submission-lecslab/{lang}.results.{num}.tsv'

        # Load the TSV file
        df = pd.read_csv(file_path, sep='\t')

        # Shift the last column down by one row
        df.iloc[:, -1] = df.iloc[:, -1].shift(1)

        # Remove the first row
        df = df.iloc[1:].reset_index(drop=True)

        # Save the modified DataFrame back to the same TSV file
        df.to_csv(file_path, sep='\t', index=False)
