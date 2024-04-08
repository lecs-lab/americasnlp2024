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

arch='attentive_lstm'
method="$1"

mkdir data/temp

for lang in bribri maya guarani 
do
  # Concat the train and augmented datasets
  # TODO: Add support for multiple aug datasets
  python scripts/conjoin_dataframes.py data/yoyodyne/$lang-train.tsv data/augmented/$lang-$method.tsv data/temp/$lang-train+$method.tsv

  yoyodyne-train \
    --experiment 2024americasnlp-$lang \
    --model_dir models/aug \
    --train data/temp/$lang-train+$method.tsv \
    --val data/yoyodyne/$lang-dev.tsv \
    --features_col 3 \
    --arch $arch \
    --batch_size 32 \
    --max_epochs 100 \
    --scheduler lineardecay \
    --log_wandb \
    --seed 0
      # --accelerator gpu \


  ckpt_file=(./models/aug/2024americasnlp-$lang/version_0/checkpoints/*.ckpt)
  ckpt_file=${ckpt_file[0]}

  echo Loading checkpoint file from $ckpt_file

  yoyodyne-predict \
    --model_dir ./models/aug \
    --experiment 2024americasnlp-$lang \
    --checkpoint "$ckpt_file" \
    --predict "data/yoyodyne/$lang-dev.tsv" \
    --output "./preds/aug/$method/$arch-$lang.tsv" \
    --features_col 3 \
    --target_col 0 \
    --arch $arch
    # --accelerator gpu \

  # Move the folder so we only ever have one numbered version
  mv ./models/aug/2024americasnlp-$lang/version_0 ./models/aug/2024americasnlp-$lang/$arch

  python ./scripts/copy_preds.py "./preds/aug/$method/$arch-$lang.tsv" "data/yoyodyne/$lang-dev.tsv"

  python ./americasnlp2024/ST2_EducationalMaterials/baseline/evaluate.py "./preds/aug/$method/$arch-$lang.tsv" > "./preds/aug/$method/$arch-$lang-eval.out"

  rm data/temp/*.tsv
done

