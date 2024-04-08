# americasnlp2024

## Checking out the submodule (shared task data)
Navigate to the root directory of this repository, then run

`git submodule init`

`git submodule update`

## Installing `yoyodyne`

Please use the forked version at https://github.com/michaelpginn/yoyodyne. 

1. You will need to use Python 3.10. You can use [pyenv](https://github.com/pyenv/pyenv) to manage versions.
2. Create a virtual environment (as some of the dependencies are nonstandard) using `venv`.
3. Navigate to the clone repo and run `pip install .`
4. Navigate back to the `americasnlp2024` repo (ensuring you're still using the right environment and Python and run scripts as needed.

## Training Scripts
Training scripts are in `scripts/`
- `train.sh` is used to train using the shared task data across all the model architectures supported by `yoyodyne`
- `train_aug.sh` is used to train using an additional augmented dataset
- `conjoin_dataframes.py` is a helper script that joins two dataframes. You can use it to combine multiple augmented datasets.
- `copy_preds.py` is a helper script used to produce the final predictions tsv file.

## Augmenting Data
Any augmented data files should be added to `data/augmented/{language}-{strategy}.tsv`. The file should not include column headers and should follow the columns Source, Target, Change.
