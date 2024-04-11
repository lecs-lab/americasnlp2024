#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00          # Max walltime              
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=americasnlp.%j.out      # Output file name
#SBATCH --error=americasnlp.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate americasnlp2024
cd "/projects/migi8081/americasnlp2024"

arch="$1"
method="$2"

mkdir data/temp

for lang in bribri guarani maya
do 
  # Concat the train/dev and augmented datasets
  # If you need multiple aug dataset, use `conjoin_dataframes` to create a joint augmented dataset and then pass that dataset here
  python scripts/conjoin_dataframes.py data/yoyodyne/$lang-train.tsv data/augmented/$lang-$method.tsv data/temp/$lang-train+$method.tsv

  rm -rf ./models/final/$method/2024americasnlp-$lang-final/*

  yoyodyne-train \
    --experiment 2024americasnlp-$lang-final \
    --model_dir models/final/$method \
    --train data/temp/$lang-train+$method.tsv \
    --val data/yoyodyne/$lang-dev.tsv \
    --features_col 3 \
    --arch $arch \
    --features_encoder_arch linear \
    --batch_size 32 \
    --max_epochs 1000 \
    --scheduler lineardecay \
    --log_wandb \
    --seed 0 \
    --accelerator gpu \


  ckpt_file=(./models/final/$method/2024americasnlp-$lang-final/version_0/checkpoints/*.ckpt)
  ckpt_file=${ckpt_file[0]}

  echo Loading checkpoint file from $ckpt_file

  yoyodyne-predict \
    --model_dir ./models/final/$method \
    --experiment 2024americasnlp-$lang-final \
    --checkpoint "$ckpt_file" \
    --predict "data/yoyodyne/$lang-test.tsv" \
    --output "./test-preds/char_$method_$arch/$lang.tsv" \
    --features_col 2 \
    --target_col 0 \
    --arch $arch \
    --accelerator gpu \

  # Move the folder so we only ever have one numbered version
  mv ./models/final/$method/2024americasnlp-$lang-final/version_0 ./models/final/$method/2024americasnlp-$lang-final/$arch

  python ./scripts/copy_preds.py "./test-preds/char_$method_$arch/$lang.tsv" "data/yoyodyne/$lang-test.tsv"
  rm data/temp/*.tsv

done 

