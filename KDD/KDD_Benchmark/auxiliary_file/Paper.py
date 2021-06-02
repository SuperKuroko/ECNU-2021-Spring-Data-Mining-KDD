#encoding: utf-8
import os
import sys
import importlib
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
importlib.reload(sys)
sys.path.append("../")
import config



if __name__ == '__main__':
    paper_file_path = os.path.join(config.DATASET_PATH,"Paper.csv")
    data = pd.read_csv(paper_file_path, encoding='utf-8')
    data = data.sort_values(by = "Title")
    data = data.reset_index(drop = True)
    print(data.head())
    vis = np.zeros(len(data))
    cnt = 0
    '''
    for i in range(1,len(data)):
        if data.loc[i]["Title"] == data.loc[i-1]["Title"]:
            if vis[i-1] == 0.0:
                print(data.loc[i-1])
                vis[i-1] = 1
                cnt += 1
            if vis[i] == 0.0:
                print(data.loc[i])
                vis[i] = 1
                cnt += 1
    print("cnt = ",cnt)
    '''
    data.to_csv("PaperWithTitleSorted.csv",index=False)
    