#!/usr/bin/env python
#encoding: utf-8
import sys
import importlib
importlib.reload(sys)
sys.path.append("../")
# sys.setdefaultencoding('utf-8')
import util
import numpy as np
import torch
import config
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
from model_trainer.make_dictionary import *


# coauthor信息
# 很多论文都有多个作者，根据paperauthor统计每一个作者的top 10(当然可以是top 20或者其他top K)的coauthor，
# 对于一个作者论文对(aid，pid),计算ID为pid的论文的作者有没有出现ID为aid的作者的top 10 coauthor中，
# (1). 可以简单计算top 10 coauthor出现的个数，
# (2). 还可以算一个得分，每个出现pid论文的top 10 coauthor可以根据他们跟aid作者的合作次数算一个分数，然后累加，
# 我简单地把coauthor和当前aid作者和合作次数作为这个coauthor出现的得分。

'''
对于四个特征函数输入参数解释:
AuthorIdPaperId:
{
    参数类型: AuthorIdPaperId类 (defined in authorIdPaperId.py)
    可以理解为唯一的input变量,其他参数在不更换源文件的情况下在每次调用时均不会改变
    该类包含(作者id,论文id,label)三个类成员
    在该文件中通过提取作者id和论文id来生成相应的特征
}

dict_coauthor
{
    参数类型: 字典 generated from coauthor.json
    key类型为作者id
    val类型仍为字典,其格式为{a1:count1,a2:count2...ak:countk}
}

dict_paperIdAuthorId_to_name_aff
{
    参数类型: 字典 generated from dictionary from paperIdAuthorId_to_name_and_affiliation.json
    key类型为string,format为"PaperId|AuthorId"的复合形式
    value类型为字典,其格式为{"affiliation":"a1|a2|...|an","name":"n1|n2|...|nm"}
}

PaperAuthor
{
    参数类型: DataFrame, read from PaperAuthor.csv by pandas
    DateFrame包括的列有:
    PaperId     int     论文编号
    AuthorId    int     论文编号
    Name        string  作者名称
    Affiliation string  隶属单位
}

Author
{
    参数类型: DataFrame, read from Author.csv by pandas
    DateFrame包括的列有:
    Id          int     作者编号
    Name        string  作者名称
    Affiliation string  隶属单位
} 
'''

# 1. 简单计算top 10 coauthor出现的个数
def coauthor_1(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
    # 从类成员中提取作者id与论文id
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId
    '''
    用xxx表示上一行的语句，以此来体现嵌套关系
    PaperAuthor["PaperId"] == int(paperId) 在某个item.PaperId = AuthorIdPaperId.paperId会返回true
    而PaperAuthor[xxx]则是枚举所有item,只保留方框内值为true的项
    xxx["AuthorId"] 则是取这些保留的item的AuthorId的值
    list(xxx.values)则是将这些值取出以列表的转换为列表
    list(map(str,xxx))是将所有值从int转换为字符串类型，再重新将map对象变回列表
    所以整个语句的含义就是返回一个列表，列表中包含了所有PaperId = AuthorIdPaperId.paperId的AuthorId,且类型为string
    '''
    curr_coauthors = list(map(str, list(PaperAuthor[PaperAuthor["PaperId"] == int(paperId)]["AuthorId"].values)))
    # 获取输入作者的最频繁的k个合作作者,以[a1,a2,a3...]的形式返回
    top_coauthors = list(dict_coauthor[authorId].keys())

    # 将两个列表转换为集合,做交集运算,计算交集的元素个数
    nums = len(set(curr_coauthors) & set(top_coauthors))
    '''
    简单分析nums的意义:实际输入的参数仅有一个Pid和一个Aid, 伴随输入的参数还有多个字典
    通过字典先找出Pid的所有作者S1
    然后又找出Aid的高频合作作者S2
    然后取交集
    也就是说，我们分析一个Pid和一个Aid之间的特征值是通过该篇论文的所有作者有多少个和给定的作者频繁合作
    蕴含的意义就是，对于作者a，如果他经常和作者b,c,d,e...k个人合作
    那么现在已知某篇论文已经被b,c,d,e...k写过了，那么很可能a也写了该论文
    nums便是上文的k
    最后返回的结果是一个Feature特征类(in feature.py)
    类的构造方法调用了util.py中的get_feature_by_list的函数,有关该函数的说明写在了util.py当中
    这里直接说明最后特征类的结果
    class Feature:
        self.name = ""
        self.dimension = 1
        self.feat_string = "1:nums"
    '''
    return util.get_feature_by_list([nums])


# 2. 还可以算一个得分，每个出现pid论文的top 10 coauthor可以根据他们跟aid作者的合作次数算一个分数，然后累加，
def coauthor_2(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
    # 从类成员中提取作者id与论文id
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    #该语句的解释同coauthor_1函数
    curr_coauthors = list(map(str, list(PaperAuthor[PaperAuthor["PaperId"] == int(paperId)]["AuthorId"].values)))

    # 与coauthor1函数不同的是，除了作者的合作作者之外，还取值他们的合作次数
    # top_coauthors的格式形如 {"name1": times1, "name2":times2...}
    top_coauthors = dict_coauthor[authorId]

    # 同样找两个对象之间的交集，与coauthor1不同的是，每个合作的作者多了一个权重:合作此时
    score = 0
    for curr_coauthor in curr_coauthors:
        if curr_coauthor in top_coauthors:
            score += top_coauthors[curr_coauthor]
    '''
    同样分析一下score的意义
    在coauthor_1函数中，曾举过例子：如果该作者与k个人合作过，而某篇论文恰好也被这k个人写过，
    那么该作者很有可能写过该论文，score在nums的基础上更精细化的处理了，考虑A与B,C均合作过
    其中A与B合作了100次，A只与C合作了1次，现在有两篇论文X,Y
    已知X被B写了，Y被C写了，那么显然A写X的概率远大于A写Y
    但是在coauthor_1函数中，(A,X)与(A,Y)却有相同的nums值，显然不太合理
    而在coauthor_2函数中 (A,X)的score = 100, (A,Y)的score = 1，区分得以体现
    返回的特征类
    class Feature:
        self.name = ""
        self.dimension = 1
        self.feat_string = "1:score"
    '''
    return util.get_feature_by_list([score])

''' String Distance Feature'''
# 1. name-a 与name1##name2##name3的距离，同理affliction-a 和 aff1##aff2##aff3的距离
def stringDistance_1(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    #首先构造出dict_paperIdAuthorId_to_name_aff的key类型: AuthorId|PaperId
    key = "%s|%s" % (paperId, authorId)

    #然后通过查询字典获取其全部的name属性与affiliation属性
    #name = "name1##name2##name3##..."
    #aff = "aff1##aff2##aff3##..."
    name = str(dict_paperIdAuthorId_to_name_aff[key]["name"])
    aff = str(dict_paperIdAuthorId_to_name_aff[key]["affiliation"])

    '''
    Author包含Id(作者编号),Name(作者名称),Affiliation(隶属单位)三个列
    Author[Author["Id"] == int(authorId)].values是所有Id为authorId的行,格式为[["Id Name Aff"]]
    list(xxx) 将其转换为 [array([Id,Name,Aff])]
    T = xxx[0]则为[Id,Name,Aff]
    因此 str(T[1])即为Name属性,str(T[2])即为Affiliation属性
    "nan"是在属性为空时的返回值，因为文件描述中有说到，属性值可能为空，因此手动将nan转换为空字符，避免干扰
    '''
    T = list(Author[Author["Id"] == int(authorId)].values)[0]
    a_name = str(T[1])
    a_aff = str(T[2])
    if a_name == "nan":
        a_name = ""
    if a_aff == "nan":
        a_aff = ""

    feat_list = []
    #需要注意的是，此时name并未被分割，因此计算的是形如 a_name与 name1##name2##name3##...之间的距离
    # 计算 a_name 与 name 的距离
    feat_list.append(len(longest_common_subsequence(a_name, name)))
    feat_list.append(len(longest_common_substring(a_name, name)))
    feat_list.append(Levenshtein_distance(a_name, name))
    # 计算 a_aff 与 aff 的距离
    feat_list.append(len(longest_common_subsequence(a_aff, aff)))
    feat_list.append(len(longest_common_substring(a_aff, aff)))
    feat_list.append(Levenshtein_distance(a_aff, aff))
    '''
    feat_list = [d1,d2,d3,d4,d5,d6]
    于是构造的特征类格式为
    class Feature:
        self.name = ""
        self.dimension = 6
        self.feat_string = "1:d1 2:d2 3:d3 4:d4 5:d5 6:d6"
    '''
    return util.get_feature_by_list(feat_list)

'''
整体分析两个字符串特征函数的意义
对于输入的Pid与Aid,查询(Pid,Aid)在PaperAuthor.csv出现的全部name与aff信息
然后又查询Author.csv中Aid对应的name与aff信息
也就是说对于一个论文作者对，分别查询(论文,作者)的姓名与隶属单位
然后又查询作者本身的姓名与隶属单位，对比二者，如果评估的数值结果较大
那么可以认为作者写了该论文。

但是该过程存在诸多问题: PaperAuthor.csv中记录的是某作者写了某论文
也就是说(Pid,Aid)不一定出现在了该文件中，查询字典的时候可能报错
其次，如果(Pid,Aid)出现在了该文件中，那就说明作者写了论文，但在介绍文件的时候说过
该文件包含噪声，出现并不意味着绝对写过，论文-作者对本身可能是错误数据
因此字符串特征函数的用意应该是消除噪音，试想一下
假设(P1,A1)出现在了PaperAuthor.csv当中3次
但是我们需要判定(P1,A1)本身是否是正确的(例如将A2抄成了A1)
那么我们对比A1在Author.csv中的姓名和单位
如果二者相同或者在语义上很接近，那么可以判定这是一条正确的记录
也就是说，这两个函数不适合作为特征函数传入训练文件当中
而是更适合作为清洗数据的预处理函数
'''
# 2. name-a分别与name1，name2，name3的距离，然后取平均，同理affliction-a和,aff1，aff2，aff3的平均距离
def stringDistance_2(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    key = "%s|%s" % (paperId, authorId)
    name = str(dict_paperIdAuthorId_to_name_aff[key]["name"])
    aff = str(dict_paperIdAuthorId_to_name_aff[key]["affiliation"])

    T = list(Author[Author["Id"] == int(authorId)].values)[0]
    a_name = str(T[1])
    a_aff = str(T[2])
    if a_name == "nan":
        a_name = ""
    if a_aff == "nan":
        a_aff = ""

    feat_list = []
    #以上均与stringDistance_1函数的内容相同

    #不同的是，将 name1##name2##...进行分割了
    # 计算 a_name 与 name 的距离
    lcs_distance = []
    lss_distance = []
    lev_distance = []
    for _name in name.split("##"):
        lcs_distance.append(len(longest_common_subsequence(a_name, _name)))
        lss_distance.append(len(longest_common_substring(a_name, _name)))
        lev_distance.append(Levenshtein_distance(a_name, _name))

    #用分割后的多个值的平均值取代整体计算结果
    feat_list += [np.mean(lcs_distance), np.mean(lss_distance), np.mean(lev_distance)]

    # 计算 a_aff 与 aff 的距离
    lcs_distance = []
    lss_distance = []
    lev_distance = []
    for _aff in aff.split("##"):
        lcs_distance.append(len(longest_common_subsequence(a_aff, _aff)))
        lss_distance.append(len(longest_common_substring(a_aff, _aff)))
        lev_distance.append(Levenshtein_distance(a_aff, _aff))

    feat_list += [np.mean(lcs_distance), np.mean(lss_distance), np.mean(lev_distance)]

    '''
    feat_list = [d1,d2,d3,d4,d5,d6]
    class Feature:
        self.name = ""
        self.dimension = 6
        self.feat_string = "1:d1 2:d2 3:d3 4:d4 5:d5 6:d6"
    '''
    return util.get_feature_by_list(feat_list)



''' 一些距离计算方法 '''

# 最长公共子序列（LCS）, 获取是a, b的最长公共子序列, 可参考下方链接
# https://leetcode-cn.com/problems/longest-common-subsequence/solution/zui-chang-gong-gong-zi-xu-lie-by-leetcod-y7u0/
def longest_common_subsequence(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            x -= 1
            y -= 1
    return result


# 最长公共子串(LSS)，对于该问题的详解可参考下方链接
# https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/solution/zui-chang-zhong-fu-zi-shu-zu-by-leetcode-solution/
def longest_common_substring(a, b):
    m = [[0] * (1 + len(b)) for i in range(1 + len(a))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(a)):
        for y in range(1, 1 + len(b)):
            if a[x - 1] == b[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return a[x_longest - longest: x_longest]


# 编辑距离,对于该问题的详解可参考下方
# https://leetcode-cn.com/problems/edit-distance/solution/bian-ji-ju-chi-by-leetcode-solution/
def Levenshtein_distance(input_x, input_y):
    xlen = len(input_x) + 1  # 此处需要多开辟一个元素存储最后一轮的计算结果
    ylen = len(input_y) + 1

    dp = np.zeros(shape=(xlen, ylen), dtype=int)
    for i in range(0, xlen):
        dp[i][0] = i
    for j in range(0, ylen):
        dp[0][j] = j

    for i in range(1, xlen):
        for j in range(1, ylen):
            if input_x[i - 1] == input_y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[xlen - 1][ylen - 1]

def conference_similarity(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
    # 从类成员中提取作者id与论文id
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId
    
    # 搜索该作者以前写过的全部论文
    all_papers = list(map(str, list(PaperAuthor[PaperAuthor["AuthorId"] == int(authorId)]["PaperId"].values)))
    
    # 搜索输入论文对应的会议id
    conferenceId = list(map(str, list(Paper[Paper["Id"] == int(paperId)]["ConferenceId"].values)))
    if len(conferenceId) == 0:
        return util.get_feature_by_list([0])
    
    # 搜索该会议id对应的FullName
    fullName = list(map(str, list(Conference[Conference["Id"] == int(conferenceId[0])]["FullName"].values)))
    if len(fullName) == 0:
        return util.get_feature_by_list([0])
    
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
        cid = list(map(str, list(Paper[Paper["Id"] == int(pid)]["ConferenceId"].values)))
        if len(cid) == 0:
            continue

        name = list(map(str, list(Conference[Conference["Id"] == int(cid[0])]["FullName"].values)))
        if len(name) == 0:
            continue
        cnt += 1
        word_vector_y = torch.zeros(1,dimension)
        name = get_origin_word_from_sentence(name[0])
        for word in name:
            v = embeds(torch.LongTensor([conference_dict[word]]))
            word_vector_y.add_(v)
        word_vector_y /= len(name)
        cs = cosine_similarity(word_vector_x,word_vector_y)
        if abs(cs) > 0.05: #阈值
            result += cs
    result /= cnt
    return util.get_feature_by_list([result])
    
def journal_similarity(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
    # 从类成员中提取作者id与论文id
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId
    
    # 搜索该作者以前写过的全部论文
    all_papers = list(map(str, list(PaperAuthor[PaperAuthor["AuthorId"] == int(authorId)]["PaperId"].values)))
    
    # 搜索输入论文对应的期刊id
    journalId = list(map(str, list(Paper[Paper["Id"] == int(paperId)]["JournalId"].values)))
    if len(journalId) == 0:
        return util.get_feature_by_list([0])
    
    # 搜索该会议id对应的FullName
    fullName = list(map(str, list(Journal[Journal["Id"] == int(journalId[0])]["FullName"].values)))
    if len(fullName) == 0:
        return util.get_feature_by_list([0])
    
    # 读取字典
    journal_dict = Read_dict_from_txt(config.JOURNAL_DICT)
    
    # 词向量维度
    dimension = 200

    # 初始化embeds
    embeds = torch.nn.Embedding(len(journal_dict),dimension)

    word_vector_x = torch.zeros(1,dimension)
    fullName = get_origin_word_from_sentence(fullName[0])
    for word in fullName:
        v = embeds(torch.LongTensor([journal_dict[word]]))
        word_vector_x.add_(v)
    word_vector_x /= len(fullName)

    result = 0
    cnt = 0
    for pid in all_papers:
        jid = list(map(str, list(Paper[Paper["Id"] == int(pid)]["JournalId"].values)))
        if len(jid) == 0:
            continue

        name = list(map(str, list(Journal[Journal["Id"] == int(jid[0])]["FullName"].values)))
        if len(name) == 0:
            continue
        cnt += 1
        word_vector_y = torch.zeros(1,dimension)
        name = get_origin_word_from_sentence(name[0])
        for word in name:
            v = embeds(torch.LongTensor([journal_dict[word]]))
            word_vector_y.add_(v)
        word_vector_y /= len(name)
        cs = cosine_similarity(word_vector_x,word_vector_y)
        if abs(cs) > 0.05: #阈值
            result += cs
    result /= cnt
    return util.get_feature_by_list([result])

def tf_idf(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
    # 从类成员中提取作者id与论文id
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId
    
    # 搜索该作者以前写过的全部论文
    all_papers = list(map(str, list(PaperAuthor[PaperAuthor["AuthorId"] == int(authorId)]["PaperId"].values)))

    # 搜索输入论文对应的关键词
    keyword = list(map(str, list(Paper[Paper["Id"] == int(paperId)]["Keyword"].values)))

    # 信息缺失
    if len(keyword) == 0 or (len(keyword) == 1 and keyword[0] == "nan"):
        return util.get_feature_by_list([0])
    keyword = get_origin_word_from_sentence(keyword[0])


    key_dict = []
    for pid in all_papers:
        key = list(map(str, list(Paper[Paper["Id"] == int(pid)]["Keyword"].values)))
        if len(key) == 0 or (len(key) == 1 and key[0] == "nan"):
            continue
        key_dict.append(get_origin_word_from_sentence(key[0]))

    if(len(key_dict) == 0):
        return util.get_feature_by_list([0])

    corpus=TextCollection(key_dict)  #构建语料库

    result = 0
    for word in keyword:
        result += corpus.tf_idf(word,corpus)
    
    return util.get_feature_by_list([result])
        

def journal_conference(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
    # 从类成员中提取作者id与论文id
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId
    
    # 搜索该作者以前写过的全部论文
    all_papers = list(map(str, list(PaperAuthor[PaperAuthor["AuthorId"] == int(authorId)]["PaperId"].values)))
    
    # 读取字典
    conference_dict = Read_dict_from_txt(config.CONFERENCE_DICT)
    journal_dict = Read_dict_from_txt(config.JOURNAL_DICT)

    # 词向量维度
    dimension = 200

    result = 0

    # 搜索输入论文对应的会议id
    conferenceId = list(map(str, list(Paper[Paper["Id"] == int(paperId)]["ConferenceId"].values)))
    if len(conferenceId) != 0:
        # 搜索该会议id对应的FullName
        c_fullName = list(map(str, list(Conference[Conference["Id"] == int(conferenceId[0])]["FullName"].values)))
        if len(c_fullName) != 0:
            # 初始化embeds
            c_embeds = torch.nn.Embedding(len(conference_dict),dimension)

            word_vector_x = torch.zeros(1,dimension)
            c_fullName = get_origin_word_from_sentence(c_fullName[0])
            for word in c_fullName:
                v = c_embeds(torch.LongTensor([conference_dict[word]]))
                word_vector_x.add_(v)
            word_vector_x /= len(c_fullName)

            c_result = 0
            c_cnt = 0
            for pid in all_papers:
                cid = list(map(str, list(Paper[Paper["Id"] == int(pid)]["ConferenceId"].values)))
                if len(cid) == 0:
                    continue

                name = list(map(str, list(Conference[Conference["Id"] == int(cid[0])]["FullName"].values)))
                if len(name) == 0:
                    continue
                c_cnt += 1
                word_vector_y = torch.zeros(1,dimension)
                name = get_origin_word_from_sentence(name[0])
                for word in name:
                    v = c_embeds(torch.LongTensor([conference_dict[word]]))
                    word_vector_y.add_(v)
                word_vector_y /= len(name)
                cs = cosine_similarity(word_vector_x,word_vector_y)
                if abs(cs) > 0.05: #阈值
                    c_result += cs
            c_result /= c_cnt
            result += c_result
    
    
    # 搜索输入论文对应的期刊id
    journalId = list(map(str, list(Paper[Paper["Id"] == int(paperId)]["JournalId"].values)))
    if len(journalId) != 0:
    
        # 搜索该会议id对应的FullName
        j_fullName = list(map(str, list(Journal[Journal["Id"] == int(journalId[0])]["FullName"].values)))
        if len(j_fullName) != 0:

            # 初始化embeds
            j_embeds = torch.nn.Embedding(len(journal_dict),dimension)

            word_vector_x = torch.zeros(1,dimension)
            j_fullName = get_origin_word_from_sentence(j_fullName[0])
            for word in j_fullName:
                v = j_embeds(torch.LongTensor([journal_dict[word]]))
                word_vector_x.add_(v)
            word_vector_x /= len(j_fullName)

            j_result = 0
            j_cnt = 0
            for pid in all_papers:
                jid = list(map(str, list(Paper[Paper["Id"] == int(pid)]["JournalId"].values)))
                if len(jid) == 0:
                    continue

                name = list(map(str, list(Journal[Journal["Id"] == int(jid[0])]["FullName"].values)))
                if len(name) == 0:
                    continue
                j_cnt += 1
                word_vector_y = torch.zeros(1,dimension)
                name = get_origin_word_from_sentence(name[0])
                for word in name:
                    v = j_embeds(torch.LongTensor([journal_dict[word]]))
                    word_vector_y.add_(v)
                word_vector_y /= len(name)
                cs = cosine_similarity(word_vector_x,word_vector_y)
                if abs(cs) > 0.05: #阈值
                    j_result += cs
            j_result /= j_cnt
            result += j_result
    return util.get_feature_by_list([result])

if __name__ == '__main__':
    pass


