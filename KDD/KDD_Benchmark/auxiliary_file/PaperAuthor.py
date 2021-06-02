#encoding: utf-8
import os
import sys
import importlib
from pandas import Series, DataFrame
import pandas as pd
importlib.reload(sys)
sys.path.append("../")
import config

if __name__ == '__main__':
    author_file_path = os.path.join(config.DATASET_PATH,"PaperAuthor.csv")
    PaperAuthor = pd.read_csv(author_file_path, encoding='utf-8')
    while True:
        x = input()
        if x == "exit":
            break
        curr_coauthors = list(map(str, list(PaperAuthor[PaperAuthor["PaperId"] == int(x)]["AuthorId"].values)))
        print(curr_coauthors)

    