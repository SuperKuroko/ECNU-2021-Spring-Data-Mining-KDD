import sys
import importlib
importlib.reload(sys)
sys.path.append("../")
# sys.setdefaultencoding('utf-8')
import util
import numpy as np
import pandas as pd
import torch
import config
from model_trainer.make_dictionary import *

def conference_similarity1(aid, pid, PaperAuthor, Paper, Conference):
    # 从类成员中提取作者id与论文id
    authorId = aid
    paperId = pid
    
    # 搜索该作者以前写过的全部论文
    all_papers = list(map(str, list(PaperAuthor[PaperAuthor["AuthorId"] == int(authorId)]["PaperId"].values)))
    print("all_papers = ",all_papers)

    # 搜索输入论文对应的会议id
    conferenceId = list(map(str, list(Paper[Paper["Id"] == int(paperId)]["ConferenceId"].values)))[0]
    
    # 搜索该会议id对应的FullName
    fullName = list(map(str, list(Conference[Conference["Id"] == int(conferenceId)]["FullName"].values)))
    if len(fullName) == 0:
        return -1

    print(fullName)
    # 读取字典
    conference_dict = Read_dict_from_txt(config.CONFERENCE_DICT)
    
    # 词向量维度
    dimension = 200

    # 初始化embeds
    embeds = torch.nn.Embedding(len(conference_dict),dimension)

    word_vector_x = torch.zeros(1,dimension)
    fullName = get_origin_word_from_sentence(fullName[0])
    for word in fullName:
        v = embeds(torch.LongTensor([conference_dict[word]]))
        word_vector_x.add_(v)
    word_vector_x /= len(fullName)

    result = 0
    cnt = 0
    for pid in all_papers:
        cid = list(map(str, list(Paper[Paper["Id"] == int(pid)]["ConferenceId"].values)))[0]
        name = list(map(str, list(Conference[Conference["Id"] == int(cid)]["FullName"].values)))
        if len(name) == 0:
            continue
        cnt += 1
        word_vector_y = torch.zeros(1,dimension)
        print("name = ",name)
        name = get_origin_word_from_sentence(name[0])
        for word in name:
            v = embeds(torch.LongTensor([conference_dict[word]]))
            word_vector_y.add_(v)
        word_vector_y /= len(name)
        cs = cosine_similarity(word_vector_x,word_vector_y)
        if abs(cs) > 0.05: #阈值
            result += cs
    result /= cnt
    return result

if __name__ == '__main__':
    PaperAuthor = pd.read_csv(config.PAPERAUTHOR_FILE)  # 加载 PaperAuthor.csv 数据
    Paper = pd.read_csv(config.PAPER_FILE)              # 加载 Paper.csv 数据
    Conference = pd.read_csv(config.CONFERENCE_FILE)    # 加载 Conference.csv 数据
    print(conference_similarity1("1534492","319590",PaperAuthor,Paper,Conference))