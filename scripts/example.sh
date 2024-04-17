#!/bin/bash


arch=attentive_lstm
method=identityexternal

mkdir data/temp

for lang in guarani
do 
  # Concat the train/dev and augmented datasets
  # If you need multiple aug dataset, use `conjoin_dataframes` to create a joint augmented dataset and then pass that dataset here
  python scripts/conjoin_dataframes.py data/yoyodyne/$lang-train.tsv data/augmented/$lang-$method.tsv data/temp/$lang-train+$method.tsv

  rm -rf ./models/test/$method-$arch/2024americasnlp-$lang-test/*


  yoyodyne-train \
    --experiment 2024americasnlp-$lang-test \
    --model_dir models/test/$method-$arch \
    --train data/temp/$lang-train+$method.tsv \
    --val data/yoyodyne/$lang-dev.tsv \
    --features_col 3 \
    --arch $arch \
    --features_encoder_arch linear \
    --batch_size 32 \
    --embedding 256 \
    --decoder_layers 1 \
    --hidden_size 1024 \
    --source_attention_heads 1 \
    --teacher_forcing \
    --max_source_length 200 \
    --max_target_length 200 \
    --max_epochs 5 \
    --scheduler lineardecay \
    --seed 0 \


  ckpt_file=(./models/test/$method-$arch/2024americasnlp-$lang-test/version_0/checkpoints/*.ckpt)
  ckpt_file=${ckpt_file[0]}

  echo Loading checkpoint file from $ckpt_file

  yoyodyne-predict \
    --model_dir ./models/test/$method-$arch \
    --experiment 2024americasnlp-$lang-test \
    --checkpoint "$ckpt_file" \
    --predict "data/yoyodyne/$lang-dev.tsv" \
    --output "./local-test-preds/char-$method-$arch/$lang-test.tsv" \
    --features_col 3 \
    --target_col 0 \
    --arch $arch \

  python ./scripts/copy_preds.py "./local-test-preds/char-$method-$arch/$lang-dev.tsv" "americasnlp2024/ST2_EducationalMaterials/data/$lang-dev.tsv"

  yoyodyne-predict \
    --model_dir ./models/test/$method-$arch \
    --experiment 2024americasnlp-$lang-test \
    --checkpoint "$ckpt_file" \
    --predict "data/yoyodyne/$lang-test.tsv" \
    --output "./local-test-preds/char-$method-$arch/$lang-test.tsv" \
    --features_col 2 \
    --target_col 0 \
    --arch $arch \

  python ./scripts/copy_preds.py "./local-test-preds/char-$method-$arch/$lang-test.tsv" "americasnlp2024/ST2_EducationalMaterials/data/$lang-test.tsv"

  # Move the folder so we only ever have one numbered version
  mv ./models/test/$method-$arch/2024americasnlp-$lang-test/version_0 ./models/test/$method-$arch/2024americasnlp-$lang-test/$arch

  rm data/temp/*.tsv

done 

