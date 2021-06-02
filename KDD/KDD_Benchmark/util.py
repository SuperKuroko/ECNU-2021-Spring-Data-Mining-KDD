#!/usr/bin/env python
#encoding: utf-8
import csv
import os
import sys

from feature import Feature
import importlib

importlib.reload(sys)
# sys.setdefaultencoding('utf-8')

'''
将字典写入csv文件函数
fieldnames:列名,格式形如 ['first_name', 'last_name']
contents:字典本体,格式形如[{'first_name': 'Baked', 'last_name': 'Beans'}, {'first_name': 'Lovely', 'last_name': 'Spam'}]
to_file: 写入的路径与文件名
其中contents的格式为list,list中包含的对象为dict
即csv每一行的格式以字典的方式呈现,list[i]代表了第i行数据,list[i][field]代表了第i行的field域的值
'''
def write_dict_to_csv(fieldnames, contents, to_file):
    with open(to_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contents)


'''
和上一个函数相反,该函数将csv文件还原成字典列表的格式
返回的参数类型形如[{'first_name': 'Baked', 'last_name': 'Beans'}, {'last_name': 'Lovely', 'last_name': 'Spam'}]
'''
def read_dict_from_csv(in_file):
    if not os.path.exists(in_file):
        return []
    with open(in_file,"r",encoding="utf-8 ") as csvfile:
        return list(csv.DictReader(csvfile))






'''
通过列表构造一个特征类
首先将列表转换为字典，转换方式为:
L[i] -> i+1: L[i]
即对于某个列表元素，以其{下标+1}为key，以元素本身为value
构造一个字典feat_dict
然后以""为name,len(list)为dimension,feat_dict为特征字典
构造一个Feature的类
'''
def get_feature_by_list(list):
    feat_dict = {}
    for index, item in enumerate(list):
        if item != 0:
            feat_dict[index+1] = item
    return Feature("", len(list), feat_dict)

#根据单个特征值构造特征类
def get_feature_by_feat(dict, feat):
    feat_dict = {}
    if feat in dict:
        feat_dict[dict[feat]] = 1
    return Feature("", len(dict), feat_dict)

#根据多个特征值构造特征类
def get_feature_by_feat_list(dict, feat_list):
    feat_dict = {}
    for feat in feat_list:
        if feat in dict:
            feat_dict[dict[feat]] = 1
    return Feature("", len(dict), feat_dict)


''' 合并 feature_list中的所有feature '''
# feature_list = [Feature1,Feature2,Feature3...]
# 存放了若干个Feature类对象,该函数用于将多个特征类对象进行合并
# 例如存在f1.feat_string = "1:5 2:10 3:15" f2.feat_string = "1:7 2:8 3:9"
# 那么合并之后得到 mf.feat_string = "1:5 2:10 3:15 4:7 5:8 6:9"
# 即将feat_string相加,但要重设index
def mergeFeatures(feature_list, name = ""):
    # print "-"*80
    # print "\n".join([feature_file.feat_string+feature_file.name for feature_file in feature_list])
    dimension = 0
    feat_string = ""

    # 枚举每一个Feature对象
    for feature in feature_list:
        if dimension == 0:#第一个
            feat_string = feature.feat_string #赋值特征
        else:
            if feature.feat_string != "":
                #修改当前feature的index
                temp = ""
                for item in feature.feat_string.split(" "):
                    # 取原下标和特征值
                    index, value = item.split(":")
                    # 构造新索引
                    temp += " %d:%s" % (int(index)+dimension, value)
                feat_string += temp # 合并特征值
        dimension += feature.dimension # 加上维度值
    # 最后构造一个特征类,将合并的特征值去除首尾空格后(strip)赋值
    merged_feature = Feature(name, dimension, {})
    merged_feature.feat_string = feat_string.strip()
    return merged_feature


# 将example类写入特征文件中
# example_list为存放若干个example类的列表, to_file为写入路径和文件名
def write_example_list_to_file(example_list, to_file):
    with open(to_file, "w") as fout:
        fout.write("\n".join([example.content + " # " + example.comment for example in example_list]))
        # 每一行的格式为 tar key1:val1 key2:val2 ... key_n:val_n # paperId authorId


'''
arff:Attribute-Relation File Format。
arff是一个ASCII文本文件，记录了一些共享属性的实例。
此类格式的文件主要由两个部分构成,头部定义和数据区。
头部定义包含了关系名称(relation name),一些属性(attributes)和对应的类型，如：
'''
# 将example类写入arff文件
def write_example_list_to_arff_file(example_list, dimension, to_file):
    with open(to_file, "w") as fout:
        out_lines = []

        out_lines.append("@relation kdd") #arff关系名称
        out_lines.append("")
        for i in range(dimension):
            # arff属性名称 属性类型为numeric数值类型
            out_lines.append("@attribute attribution%d numeric" % (i+1))
        # 取值类型限定为{0,1}
        out_lines.append("@attribute class {0, 1}")

        out_lines.append("")
        out_lines.append("@data") # 数据区域

        for example in example_list:
            feature_list = [0.0] * dimension
            s = example.content.split(" ")
            #s = [tar,key1:val1,key2:val2,...,key_n:val_n]
            target = s[0]
            for item in s[1:]:
                if item == "": #跳过空项
                    continue
                k, v = int(item.split(":")[0]) - 1, float(item.split(":")[1])
                feature_list[k] = v
            # 即将Feature全部特征值取出按下标顺序记录在feature中
            feature = ",".join(map(str, feature_list))
            # 用逗号作为分隔符拼接起来,最后并上target作为类标
            out_lines.append("%s,%s" % (feature, target))

        fout.write("\n".join(out_lines))


if __name__ == '__main__':
    s  = "0 ".split(" ")
    print(s)