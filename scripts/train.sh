#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # Number of requested cores
#SBATCH --mem=8G
#SBATCH --time=6:00:00          # Max walltime              
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

for lang in bribri maya guarani 
do
  for arch in pointer_generator_lstm attentive_lstm transformer pointer_generator_transformer # transducer
  do
    yoyodyne-train \
      --experiment 2024americasnlp-$lang \
      --model_dir models \
      --train data/yoyodyne/$lang-train.tsv \
      --val data/yoyodyne/$lang-dev.tsv \
      --features_col 3 \
      --arch $arch \
      --features_encoder_arch linear \
      --batch_size 32 \
      --max_epochs 100 \
      --scheduler lineardecay \
      --log_wandb \
      --no_save_best \
      --seed 0 \
      --accelerator gpu \


    ckpt_file=(./models/2024americasnlp-$lang/version_0/checkpoints/*.ckpt)
    ckpt_file=${ckpt_file[0]}

    echo Loading checkpoint file from $ckpt_file

    yoyodyne-predict \
      --model_dir ./models \
      --experiment 2024americasnlp-$lang \
      --checkpoint "$ckpt_file" \
      --predict "data/yoyodyne/$lang-dev.tsv" \
      --output "./preds/$arch/$lang.tsv" \
      --features_col 3 \
      --target_col 0 \
      --accelerator gpu \
      --arch $arch

    # Move the folder so we only ever have one numbered version
    mv ./models/2024americasnlp-$lang/version_0 ./models/2024americasnlp-$lang/$arch

    python ./scripts/copy_preds.py "./preds/$arch/$lang.tsv" "data/yoyodyne/$lang-dev.tsv"

    python ./americasnlp2024/ST2_EducationalMaterials/baseline/evaluate.py "./preds/$arch/$lang.tsv" > "./preds/$arch/$lang-eval.out"
  done
done

