#!/usr/bin/env python
#encoding: utf-8
import os
'''
python在安装时，默认的编码是ascii
当程序中出现非ascii编码时，python的处理常常会报错UnicodeDecodeError:
 'ascii' codec can't decode byte 0x?? in position 1:ordinal not in range(128)，
python没办法处理非ascii编码的，此时需要自己设置python的默认编码，
一般设置为utf8的编码格式，在程序中加入以下代码：即可将编码设置为utf-8
在其他文件中也是同样的作用，便不做多余的解释
'''
import sys
import importlib
importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("../")

import util
import json
from collections import Counter
import config

# 根据PaperAuthor.csv，获取每个作者的top k个共作者
def get_top_k_coauthors(paper_author_path, k, to_file):
    #首先读取csv文件到data中
    data = util.read_dict_from_csv(paper_author_path)
    #创建字典1，将相同论文的作者进行合并，
    #其元素类型为: key = PaperId,val = [AuthorId1,AuthodId2...]
    dict_paperId_to_authors = {}
    for item in data:  #对于csv文件中的每一行，即每一项
        paperId = int(item["PaperId"])   #读取PaperId列
        authorId = int(item["AuthorId"]) #读取AuthorId列
        #先检索字典中是否存在键，没有则需创建空列表
        if paperId not in dict_paperId_to_authors:
            dict_paperId_to_authors[paperId] = []
        #然后将值append到列表中
        dict_paperId_to_authors[paperId].append(authorId)

    #然后再创建字典2，统计合作作者的次数
    #其元素类型为: key = authorId,value = {coauthorId1:c1,coauthorId2:c2...}
    #其中value是Counter()计数器类型
    dict_author_to_coauthor = {}
    #枚举字典1中的每一项，对于某篇论文
    for paperId in dict_paperId_to_authors:
        authors = dict_paperId_to_authors[paperId] #获取该论文的所有作者
        n = len(authors)   #记录作者数
        for i in range(n): #枚举所有的组合类型
            for j in range(i+1, n):
                #尚不在字典中的键，则先初始化一个Counter计数器
                if authors[i] not in dict_author_to_coauthor:
                    dict_author_to_coauthor[authors[i]] = Counter()
                if authors[j] not in dict_author_to_coauthor:
                    dict_author_to_coauthor[authors[j]] = Counter()
                #然后将相应的计数器+1
                dict_author_to_coauthor[authors[i]][authors[j]] += 1
                dict_author_to_coauthor[authors[j]][authors[i]] += 1

    print("取 top k...")
    # 取 top k
    # authorid --> { author1: 100, author2: 45}
    res = {}#字典3，结果存储字典,key = authorId, value = dict{coauthorId1:c1,coauthorId2:c2...}
    #枚举字典2的每一项
    for authorId in dict_author_to_coauthor:
        res[authorId] = {} #先初始化键值字典
        #调用Counter的most_common()函数，只将最频繁出现的k个值写入字典中
        for coauthorId, freq in dict_author_to_coauthor[authorId].most_common(k):
            res[authorId][coauthorId] = freq

    print("dump...")
    #将结果编码成json文件，写入指定路径
    json.dump(res, open(to_file, "w", encoding="utf-8"))


if __name__ == '__main__':
    k = 10  #可以在此处根据需要修改k的值
    get_top_k_coauthors(
        os.path.join(config.DATASET_PATH, "PaperAuthor.csv"),#源文件路径
        k, 
        os.path.join(config.DATA_PATH, "coauthor.json")) #生成文件路径

