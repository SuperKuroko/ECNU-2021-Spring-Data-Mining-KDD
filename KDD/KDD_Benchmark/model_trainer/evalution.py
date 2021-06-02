#!/usr/bin/env python
#encoding: utf-8
import os
import sys
import importlib
importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("../")
import util
import config
from confusion_matrix import Alphabet, ConfusionMatrix

# 对模型的预测结果，重新进行整理，得到想要的格式的预测结果
# 即通过特征文件记录的Aid与Pid和result文件中的预测结果，合并出一个.predict文件
# 使得文件格式与标准答案文件一致
def get_prediction(test_feature_path, test_result_path, to_file):
    # 去除首尾空白符之后,分行存储两个文件的每一行
    feature_list = [line.strip() for line in open(test_feature_path)]
    predict_list = [line.strip() for line in open(test_result_path)]

    dict_authorId_to_predict = {}
    for feature, predict in zip(feature_list, predict_list):
        #通过#分隔符找到pid和aid
        paperId, authorId = feature.split(" # ")[-1].split(" ")
        paperId = int(paperId)
        authorId = int(authorId)

        #初始化
        if authorId not in dict_authorId_to_predict:
            dict_authorId_to_predict[authorId] = {}
            dict_authorId_to_predict[authorId]["ConfirmedPaperIds"] = []
            dict_authorId_to_predict[authorId]["DeletedPaperIds"] = []
        
        #根据便签归类结果
        if predict == "1":
            dict_authorId_to_predict[authorId]["ConfirmedPaperIds"].append(paperId)
        if predict == "0":
            dict_authorId_to_predict[authorId]["DeletedPaperIds"].append(paperId)

    # to csv, 按照字典的key值进行排序
    items = sorted(list(dict_authorId_to_predict.items()), key=lambda x: x[0])

    data = []
    for item in items:
        AuthorId = item[0]
        # 以空格为分隔符进行字符串的拼接
        ConfirmedPaperIds = " ".join(map(str, item[1]["ConfirmedPaperIds"]))
        DeletedPaperIds = " ".join(map(str, item[1]["DeletedPaperIds"]))
        # 转换为字典
        data.append({"AuthorId": AuthorId, "ConfirmedPaperIds": ConfirmedPaperIds, "DeletedPaperIds": DeletedPaperIds})
    # 将字典写为csv文件
    util.write_dict_to_csv(["AuthorId", "ConfirmedPaperIds", "DeletedPaperIds"], data, to_file)


# 评估（预测 vs 标准答案）
def Evalution(gold_file_path, pred_file_path):
    gold_authorIdPaperId_to_label = {} #标准答案存储字典
    pred_authorIdPaperId_to_label = {} #预测结果存储字典
    #上述两字典的数据类型为 {key->tuple:val->int}
    #其中tuple = (AuthorId, paperId), int = 正样本1/负样本0

    #读取标准答案文件到gold_data中,type(gold_data)为list[dict1,dict2...]
    gold_data = util.read_dict_from_csv(gold_file_path)
    #枚举每一项
    for item in gold_data:
        AuthorId = item["AuthorId"] 
        # 正样本
        for paperId in item["ConfirmedPaperIds"].split(" "):
            gold_authorIdPaperId_to_label[(AuthorId, paperId)] = "1"
        # 负样本
        for paperId in item["DeletedPaperIds"].split(" "):
            gold_authorIdPaperId_to_label[(AuthorId, paperId)] = "0"

    #读取预测结果文件到pred_data中，流程同上
    pred_data = util.read_dict_from_csv(pred_file_path)
    for item in pred_data:
        AuthorId = item["AuthorId"]
        # 正样本
        for paperId in item["ConfirmedPaperIds"].split(" "):
            pred_authorIdPaperId_to_label[(AuthorId, paperId)] = "1"
        # 负样本
        for paperId in item["DeletedPaperIds"].split(" "):
            pred_authorIdPaperId_to_label[(AuthorId, paperId)] = "0"

    # evaluation
    alphabet = Alphabet() #创建一个Alphabet类,定义于confusion_matrix.py文件当中
    alphabet.add("0") # 添加label:0, {0:0}
    alphabet.add("1") # 添加label:1, {1:1}

    # 以alphabet为参数创建一个混淆矩阵
    cm = ConfusionMatrix(alphabet)

    # 统计每一条记录的预测结果和标准答案,加入混淆矩阵
    for AuthorId, paperId in gold_authorIdPaperId_to_label:
        gold = gold_authorIdPaperId_to_label[(AuthorId, paperId)]
        pred = pred_authorIdPaperId_to_label[(AuthorId, paperId)]
        cm.add(pred, gold)

    return cm



if __name__ == '__main__':
    gold_file_path = sys.argv[1] #从命令行参数获取标答文件路径
    pred_file_path = sys.argv[2] #从命令行参数获取预测结果路径
    #ConfusionMatrix类位于 confusion_matrix.py文件当中
    #调用Evalution函数，返回一个ConfusionMatrix类实例
    cm = Evalution(gold_file_path, pred_file_path)
    #调用ConfusionMatrix类的get_accuracy()函数
    acc = cm.get_accuracy()
    # 打印评估结果
    print("")
    print("##" * 20)
    print("    评估结果, 以Accuracy为准")
    print("##" * 20)
    print("")
    cm.print_out() #调用ConfusionMatrix类的print_out()函数
