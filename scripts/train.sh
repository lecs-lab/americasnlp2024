for lang in maya # bribri maya guarani 
do
  for arch in pointer_generator_lstm # attentive_lstm transformer transducer pointer_generator_transformer
  do
    yoyodyne-train \
      --experiment 2024americasnlp-$lang \
      --model_dir models \
      --train data/yoyodyne_format/$lang-train.tsv \
      --val data/yoyodyne_format/$lang-dev.tsv \
      --features_col 2 --features_sep , --target_col 3 \
      --arch $arch \
      --batch_size 32 \
      --max_epochs 2 \
      --scheduler lineardecay \
      --log_wandb \
      --seed 0

    ckpt_file=(./models/2024americasnlp-$lang/version_0/checkpoints/*.ckpt)
    ckpt_file=${ckpt_file[0]}

    echo Loading checkpoint file from $ckpt_file

    yoyodyne-predict \
      --model_dir ./models \
      --experiment 2024americasnlp-$lang \
      --checkpoint "$ckpt_file" \
      --predict "data/yoyodyne_format/$lang-dev.tsv" \
      --output "./preds/$arch/$lang.tsv" \
      --target_col 0 \
      --arch $arch

    # Move the folder so we only ever have one numbered version
    mv ./models/2024americasnlp-$lang/version_0 ./models/2024americasnlp-$lang/$arch

    python ./scripts/copy-preds.py "./preds/$arch/$lang.tsv" "data/yoyodyne_format/$lang-dev.tsv"

    python ./americasnlp2024/ST2_EducationalMaterials/baseline/evaluate.py "./preds/$arch/$lang.tsv" > "./preds/$arch/$lang-eval.out"
  done
done

