# encoding: utf-8
# 将特征文件的每一行表示为一个类
class Example:
    def __init__(self, target, feature, comment=""):
        # target表示样本类型 feature.feat_string表示特征值
        self.content = str(target) + " " + feature.feat_string
        self.comment = comment
