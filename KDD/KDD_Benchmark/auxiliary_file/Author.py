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
    author_file_path = os.path.join(config.DATASET_PATH,"Author.csv")
    data = pd.read_csv(author_file_path, encoding='utf-8')
    T = list(data[data["Id"] == int(input())].values)[0]
    print(str(T[2]))
    '''
    dic = {}
    for i in range (len(data)):
        id = data.loc[i]["Id"]
        if id in dic:
            dic[id].append(data.loc[i]["Name"])
        else:
            dic[id] = [data.loc[i]["Name"]]
    for e in dic:
        if len(dic[e]) > 1:
            print(e,":",dic[e])
    '''
