#coding:utf-8
# 特征类
class Feature:
    def __init__(self,name, dimension, feat_dict):
        '''
        特征类的构造函数
        传入的参数包括特征名称，特征维度以及特征字典
        对于特征字典，调用下文的featDict2String函数
        将形如{k1:v1,k2:v2,k3:v3...kn:vn}的字典首先以key进行排序
        然后转换为 "k1:v1 k2:v2 k3:v3..."的字符串形式
        '''
        self.name = name #特征名称
        self.dimension = dimension #维度
        self.feat_string = self.featDict2String(feat_dict) #特征: "3:1 7:1 10:0.5"


    def featDict2String(self, feat_dict):
        #按键值排序
        list = [str(key)+":"+str(feat_dict[key]) for key in sorted(feat_dict.keys())]
        return " ".join(list)




