from typing import Optional, List
import fire
import datasets
from transformers import MT5TokenizerFast, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, \
    MT5ForConditionalGeneration, MT5Config, PreTrainedTokenizerBase, Seq2SeqTrainer, Seq2SeqTrainingArguments, EvalPrediction
import torch
import wandb
import random
import numpy as np
import sacrebleu

MAX_INPUT_LENGTH = 64
MAX_OUTPUT_LENGTH = 64
BATCH_SIZE = 64
MAX_EPOCHS = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
print(device)


def main(mode: str = 'train',
         language: str = 'bribri',
         tokenizer_vocab_size: int = 1024,
         model_path: Optional[str] = None,
         pretrained: Optional[str] = None,
         seed: int = 0):
    """Runs the training code

    Args:
        mode (str, optional): 'train' | 'eval' | 'test'
        language (str, optional): 'bribri' | 'guarani' | 'maya'
        model_path (Optional[str], optional): Path to trained model `.pth` file, for inference.
        pretrained (Optional[str], optional): Pretrained model to finetune. If not provided, training will use randomly initialized T5 model.
        seed (int, optional): Random seed.
    """
    random.seed(seed)
    wandb.init(
        project="2024americasnlp",
        entity="lecslab",
        config={
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "experiment": "baseline",
            "language": language,
            "vocab_size": tokenizer_vocab_size,
            "pretrained": pretrained,
        }
    )

    if pretrained is not None:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained)
    else:
        tokenizer: PreTrainedTokenizerBase = MT5TokenizerFast.from_pretrained(
            f"../tokenizers/{language}_{tokenizer_vocab_size}")

    # Load data
    folder_path = f'../americasnlp2024/ST2_EducationalMaterials/data/'
    train = datasets.load_dataset("csv", data_files=folder_path + f"{language}-train.tsv", delimiter="\t")
    dev = datasets.load_dataset("csv", data_files=folder_path + f"{language}-dev.tsv", delimiter="\t")
    dataset = datasets.DatasetDict({
        'train': train['train'],
        'dev': dev['train']
    })

    # List of all "Change" tags
    all_tags = sorted(set([tag.strip() for seq in dataset['train']['Change'] for tag in seq.split(",")]))

    def encode_tags(tag_string):
        """Converts a tag string in the format CAT1:VA1L, CAT2:VAL2, etc to a list of input ids"""
        tags = [tag.strip() for tag in tag_string.split(",")]
        tag_ids = [tokenizer.vocab_size + all_tags.index(tag) for tag in tags]
        return tag_ids

    def tokenize(batch):
        inputs = tokenizer(batch['Source'], text_target=batch['Target'],
                           truncation=True, padding=False, max_length=MAX_INPUT_LENGTH)
        tag_input_ids = [encode_tags(change_string) for change_string in batch['Change']]
        batch_size = len(inputs['input_ids'])
        inputs['input_ids'] = [inputs['input_ids'][i] + tag_input_ids[i] for i in range(batch_size)]
        inputs['attention_mask'] = [inputs['attention_mask'][i] +
                                    [1] * len(tag_input_ids[i]) for i in range(batch_size)]
        return inputs

    dataset = dataset.map(tokenize, batched=True).select_columns(['input_ids', 'attention_mask', 'labels'])
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    if pretrained is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
    else:
        config = MT5Config(vocab_size=tokenizer.vocab_size + len(all_tags))
        model = MT5ForConditionalGeneration(config)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    training_args = Seq2SeqTrainingArguments(
        output_dir="../models",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=MAX_EPOCHS,
        predict_with_generate=True,
    )

    def compute_metrics(eval_preds: EvalPrediction):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("Preds:", decoded_preds[:10])
        print("Labels:", decoded_labels[:10])

        accuracy = (
            sum([int(r == p) for r, p in zip(decoded_labels, decoded_preds)]) / len(decoded_labels) * 100
        )
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score
        return {"accuracy": accuracy, "bleu": bleu, "chrf": chrf}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
