
import os
import pympi
import random
import pandas as pd


random.seed(2024)

DATA_PATH = "../data/extra_monolingual/doreco_yuca1254_core/"

def main():
    files = [file for file in os.listdir(DATA_PATH) if file.endswith(".eaf")]

    all_maya_data = []
    for file in files:
        eafob = pympi.Elan.Eaf(DATA_PATH + file)

        tx_tier = [tier for tier in eafob.get_tier_names() if tier.startswith("tx")][0]

        for annotation in eafob.get_annotation_data_for_tier(tx_tier):
            utterance = annotation[2]
            all_maya_data.append(utterance)



    all_maya_data = [x.strip().capitalize() for x in all_maya_data]

    # some simple filtering of stuff in the annotation scheme
    all_maya_data = [x for x in all_maya_data if "<p" not in x and "**" not in x]

    print(len(all_maya_data))

    return
    # make into some DF
    for sample_size in [10, 50, 100, 250, 500, 1000, 1500, 2000]:
        sampled = random.sample(all_maya_data, sample_size)
        df_dict = {"Source": sampled, "Target": sampled, "Change": ["NOCHANGE"] * len(sampled)}

        df_to_save = pd.DataFrame.from_dict(df_dict)
        df_to_save.to_csv(f"../data/augmented/maya-identity-external_{sample_size}.tsv", sep='\t', header=False,
                          index=False)


    return



if __name__ == "__main__":
    main()