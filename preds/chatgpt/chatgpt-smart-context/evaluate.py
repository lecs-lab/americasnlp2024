from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
import pandas as pd

def load_data(source_path):
    source_df = pd.read_csv(source_path, sep='\t', header=0)

    return source_df["Target"], source_df["Predicted Target"]


def calc_bleu(y_trues, y_preds):

    bleu_scores = []

    for i in range(len(y_trues)):
        y_true = y_trues[i]
        y_pred = y_preds[i]
        bs = sentence_bleu([y_true], y_pred)
        #print(y_true,"|", y_pred, "|", bs)
        bleu_scores.append(bs)

    return sum(bleu_scores) / len(bleu_scores) 

def calc_accuracy(y_true, y_pred):
    return accuracy_score(y_true,y_pred)
    

def calc_chrf(y_trues,y_preds):
    chrf_scores = []

    for i in range(len(y_trues)):
        y_true = y_trues[i]
        y_pred = y_preds[i]
        bs = sentence_chrf(y_true.split(), y_pred.split())
        #print(y_true,"|", y_pred, "|", bs)
        chrf_scores.append(bs)

    return sum(chrf_scores) / len(chrf_scores) 




if __name__ == '__main__':
    print("start")
    location = ""
    langs = ["bribri", "guarani", "maya"]
    file_sufix = "-dev-preds-smart.tsv"
    for lang in langs:
        print(lang, ":")
        source_path = location + lang + file_sufix
        y_true, y_pred = load_data(source_path)
        bleu_score = calc_bleu(y_true,y_pred)
        acc = calc_accuracy(y_true,y_pred)
        chrf = calc_chrf(y_true,y_pred)
        print("Accuracy", acc * 100)
        print("BLEU", bleu_score * 100)
        print("chrf_score", chrf * 100)




