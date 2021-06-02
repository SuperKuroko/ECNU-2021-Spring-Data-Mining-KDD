#!/usr/bin/env python
#encoding: utf-8
import os
import socket
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

# 当前工作目录,最后指向KDD_benchmark即可
#CWD = "/home/username/KDD/KDD_benchmark" # Linux系统目录
CWD = r"C:\Users\Kuroko\Data-Mining\KDD\KDD_Benchmark" # Windows系统目录

DATA_PATH = os.path.join(CWD, "data")  #~\KDD_Benchmark\data
DATASET_PATH = os.path.join(DATA_PATH, "dataset") #~\KDD_Benchmark\data\dataset

# 训练和测试文件（训练阶段有验证数据，测试阶段使用测试数据
#~\KDD_Benchmark\data\dataset\train_set\Train.csv
#~\KDD_Benchmark\data\dataset\valid_set\Valid.csv
#~\KDD_Benchmark\data\dataset\valid_set\Valid.gold.csv
TRAIN_FILE = os.path.join(DATASET_PATH, "train_set", "Train.csv")
#TEST_FILE = os.path.join(DATASET_PATH, "test_set", "Test.01.csv")
TEST_FILE = os.path.join(DATASET_PATH, "valid_set", "Valid.csv")
GOLD_FILE = os.path.join(DATASET_PATH, "valid_set", "Valid.gold.csv")

# 模型文件, ~\KDD_Benchmark\model\kdd.model
MODEL_PATH = os.path.join(CWD, "model", "kdd.model") 

# 训练和测试特征文件, ~\KDD_Benchmark\feature\train.feature&test.feature
TRAIN_FEATURE_PATH = os.path.join(CWD, "feature", "train.feature")
TEST_FEATURE_PATH = os.path.join(CWD, "feature", "test.feature")

# 分类在测试集上的预测结果,~\KDD_Benchmark\predict\test.result
TEST_RESULT_PATH = os.path.join(CWD, "predict", "test.result")
# 重新格式化的预测结果, ~\KDD_Benchmark\predict\test.predict
TEST_PREDICT_PATH = os.path.join(CWD, "predict", "test.predict")

#共作者数据文件, ~\KDD_Benchmark\data\dataset\coauthor.json
COAUTHOR_FILE = os.path.join(DATASET_PATH, "coauthor.json")

#姓名和单位的整合后文件 ~\KDD_Benchmark\data\dataset\paperIdAuthorId_to_name_and_affiliation.json
PAPERIDAUTHORID_TO_NAME_AND_AFFILIATION_FILE = os.path.join(DATASET_PATH, "paperIdAuthorId_to_name_and_affiliation.json")

#论文-作者对文件 ~\KDD_Benchmark\data\dataset\PaperAuthor.csv
PAPERAUTHOR_FILE = os.path.join(DATASET_PATH, "PaperAuthor.csv")

#作者信息文件 ~\KDD_Benchmark\data\dataset\Author.csv
AUTHOR_FILE = os.path.join(DATASET_PATH, "Author.csv")

#论文信息文件 ~\KDD_Benchmark\data\dataset\Paper.csv
PAPER_FILE = os.path.join(DATASET_PATH, "Paper.csv")

#杂志信息文件 ~\KDD_Benchmark\data\dataset\Paper.csv
JOURNAL_FILE = os.path.join(DATASET_PATH, "Journal.csv")

#会议信息文件 ~\KDD_Benchmark\data\dataset\Paper.csv
CONFERENCE_FILE = os.path.join(DATASET_PATH, "Conference.csv")

#会议信息文件 ~\KDD_Benchmark\data\dataset\Journal_dict.txt
JOURNAL_DICT = os.path.join(DATASET_PATH, "Journal_dict.txt")

#会议信息文件 ~\KDD_Benchmark\data\dataset\Journal_dict.txt
CONFERENCE_DICT = os.path.join(DATASET_PATH, "Conference_dict.txt")

