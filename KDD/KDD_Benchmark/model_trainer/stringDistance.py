#!/usr/bin/env python
#encoding: utf-8
import os
import sys
import importlib
importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("../")
import config
import json
import util

# PaperAuthor.csv文件包含噪音的，同一个(authorid,paperid)可能出现多次，
# 则可以把同一个(authorid,paperid)对的多个name和多个affiliation合并起来。例如
# aid,pid,name1,aff1
# aid,pid,name2,aff2
# aid,pid,name3,aff3
# 得到aid,pid,name1##name2##name3,aff1##aff2##aff3,“##”为分隔符
def load_paperIdAuthorId_to_name_and_affiliation(PaperAuthor_PATH, to_file):
    d = {} #字典1
    #KDD_Benchmark\data\dataset\PaperAuthor.csv, with format:
    '''
        PaperId->   int->论文编号
       AuthorId->   int->作者编号
           Name->string->作者名称
    Affiliation->string->隶属单位
    '''
    data = util.read_dict_from_csv(PaperAuthor_PATH)#读取PaperAuthor.csv文件
    for item in data: #枚举每一条记录
        PaperId = item["PaperId"]
        AuthorId = item["AuthorId"]
        Name = item["Name"]
        Affiliation = item["Affiliation"]
        #提取各个参数
        key = "%s|%s" % (PaperId, AuthorId) #构造字典key值
        if key not in d: #对于尚未在字典中的,初始化value为{Name:[],Affiliation:[]}
            d[key] = {}
            d[key]["Name"] = []
            d[key]["Affiliation"] = []
        if Name != "":   #空项不并入字典1中,下同
            d[key]["Name"].append(Name)
        if Affiliation != "":
            d[key]["Affiliation"].append(Affiliation)

    t = {} #字典2
    for key in d: #对于字典1中的每一项
        name = "##".join(d[key]["Name"]) #将val列表中的每一项用##连接为字符串
        affiliation = "##".join(d[key]["Affiliation"])
        #重构字典
        t[key] = {}
        t[key]["name"] = name
        t[key]["affiliation"] = affiliation
    #通过dump函数写入paperIdAuthorId_to_name_and_affiliation.json文件当中
    json.dump(t, open(to_file, "w", encoding="utf-8"))

if __name__ == '__main__':
    load_paperIdAuthorId_to_name_and_affiliation(config.PAPERAUTHOR_FILE, config.DATASET_PATH + "/paperIdAuthorId_to_name_and_affiliation.json")


