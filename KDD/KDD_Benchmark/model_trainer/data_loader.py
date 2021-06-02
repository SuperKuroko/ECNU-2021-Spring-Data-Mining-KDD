#!/usr/bin/env python
#encoding: utf-8
# import sys
# sys.setdefaultencoding('utf-8')
from authorIdPaperId import AuthorIdPaperId
import util

# 加载训练数据
def load_train_data(train_path):
    #加载训练集Train.csv
    data = util.read_dict_from_csv(train_path)
    authorIdPaperIds = [] #存储AuthorIdPaperId类的列表
    #枚举训练集的每一项(AuthorId,ConfirmedPaperIds,DeletedPaperIds)
    for item in data:
        authorId = item["AuthorId"] #取AuthorId列的值
        # 构造训练正样本
        for paperId in item["ConfirmedPaperIds"].split(" "):
            #调用AuthorIdPaperId类的构造函数
            authorIdPaperId = AuthorIdPaperId(authorId, paperId)
            authorIdPaperId.label = 1  #正样本类标
            authorIdPaperIds.append(authorIdPaperId) #存入列表

        # 构造训练负样本，同上
        for paperId in item["DeletedPaperIds"].split(" "):
            authorIdPaperId = AuthorIdPaperId(authorId, paperId)
            authorIdPaperId.label = 0  # 负样本类标
            authorIdPaperIds.append(authorIdPaperId)
    return authorIdPaperIds #返回列表集


def load_test_data(test_path): 
    #加载测试集Test.csv/验证集Valid.csv
    data = util.read_dict_from_csv(test_path)
    authorIdPaperIds = []
    for item in data:
        authorId = item["AuthorId"]
        # 构造测试样本，将所有的PaperIds分隔构造类并全部附加进列表中
        for paperId in item["PaperIds"].split(" "):
            authorIdPaperId = AuthorIdPaperId(authorId, paperId)
            authorIdPaperId.label = -1  # 待预测，暂时赋值为1...
            authorIdPaperIds.append(authorIdPaperId)
    return authorIdPaperIds

