#encoding: utf-8
import sys
sys.path.append("../")
import json
import pandas
from model_trainer.evalution import get_prediction, Evalution
from model_trainer.data_loader import load_train_data
from model_trainer.data_loader import load_test_data
from model_trainer.make_feature_file import Make_feature_file
from feature_functions import *
from classifier import *


class Trainer(object):
    def __init__(self,
                classifier, # 分类器
                model_path, # 模型路径
                feature_function_list, # 特征函数列表
                train_feature_path, # 训练集的特征文件路径
                test_feature_path,  # 测试集的特征文件路径
                test_result_path):  # 测试集的结果文件路径

        self.classifier = classifier
        self.model_path = model_path
        self.feature_function_list = feature_function_list
        self.train_feature_path = train_feature_path
        self.test_feature_path = test_feature_path
        self.test_result_path = test_result_path

    # 生成特征文件,调用make_feature_file中的Make_feature_file函数
    '''
    train_AuthorIdPaperIds: 训练集 KDD_Benchmark\data\dataset\train_set\Train.csv
    test_AuthorIdPaperIds:  测试集 KDD_Benchmark\data\dataset\valid_set\Valid.csv
    dict_coauthor:          字典,从coauthor.json读取生成
    dict_paperIdAuthorId_to_name_aff: 字典,从paperIdAuthorId_to_name_and_affiliation.json读取生成
    PaperAuthor:            DataFrame,从PaperAuthor.csv中得到
    Author:                 DataFrame,从Author.csv中得到
    '''
    def make_feature_file(self, train_AuthorIdPaperIds, test_AuthorIdPaperIds, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference):
        # 打印提示字段
        print(("-"*120))
        print(("\n".join([f.__name__ for f in feature_function_list])))
        print(("-" * 120))

        print("make train feature file ...")
        # 生成KDD_Benchmark\feature\train.feature 和 KDD_Benchmark\feature\train.feature.arff
        Make_feature_file(train_AuthorIdPaperIds, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference, self.feature_function_list, self.train_feature_path)
        print("make test feature file ...")
        # 生成KDD_Benchmark\feature\test.feature 和 KDD_Benchmark\feature\test.feature.arff
        Make_feature_file(test_AuthorIdPaperIds, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference, self.feature_function_list, self.test_feature_path)

    # 通过特征文件和sklearn的库训练模型
    def train_mode(self):
        self.classifier.train_model(self.train_feature_path, self.model_path)

    # 用训练好的模型和测试文件生成预测结果 
    def test_model(self):
        self.classifier.test_model(self.test_feature_path, self.model_path, self.test_result_path)




# 最终测试的主函数流程
if __name__ == "__main__": 

    ''' 特征函数列表 '''
    feature_function_list = [
        #coauthor_1,
        #coauthor_2,
        #stringDistance_1,
        #stringDistance_2,
        #conference_similarity,
        #journal_similarity,
        tf_idf,
        #journal_conference,
    ]

    ''' 分类器 '''
    # 决策树，NB，等
    #classifier = Classifier(skLearn_DecisionTree())           #决策树
    #classifier = Classifier(skLearn_NaiveBayes())             #朴素贝叶斯
    #classifier = Classifier(skLearn_lr())                     #逻辑回归
    #classifier = Classifier(sklearn_RandomForestClassifier()) #随机森林


    #classifier = Classifier(skLearn_svm())                    #支持向量机
    #classifier = Classifier(skLearn_KNN())                    #k近邻算法(default k = 3)
    classifier = Classifier(skLearn_AdaBoostClassifier())     #集成学习
    #classifier = Classifier(sklearn_VotingClassifier())       #投票分类(hard)

    ''' model path '''
    model_path = config.MODEL_PATH

    ''' train feature_file & test feature_file & test result path '''
    train_feature_path = config.TRAIN_FEATURE_PATH
    test_feature_path = config.TEST_FEATURE_PATH
    test_result_path = config.TEST_RESULT_PATH

    ''' Trainer '''
    trainer = Trainer(classifier, model_path, feature_function_list, train_feature_path, test_feature_path, test_result_path)

    ''' load data '''
    print("loading data...")
    train_AuthorIdPaperIds = load_train_data(config.TRAIN_FILE)  # 加载训练数据,Train.csv
    test_AuthorIdPaperIds = load_test_data(config.TEST_FILE)  # 加载测试数据,Valid.csv/Test.csv
    # coauthor.json, 共作者数据, 
    dict_coauthor = json.load(open(config.COAUTHOR_FILE,"r",encoding="utf-8"))#, encoding="utf-8"

    # paperIdAuthorId_to_name_and_affiliation.json,姓名和单位的复合数据
    # (paperId, AuthorId) --> {"name": "name1##name2", "affiliation": "aff1##aff2"}
    dict_paperIdAuthorId_to_name_aff \
        = json.load(open(config.PAPERIDAUTHORID_TO_NAME_AND_AFFILIATION_FILE,"r",encoding="utf-8"))#, encoding="utf-8"
    
    # 使用pandas加载csv数据
    PaperAuthor = pandas.read_csv(config.PAPERAUTHOR_FILE)  # 加载 PaperAuthor.csv 数据
    Author = pandas.read_csv(config.AUTHOR_FILE)            # 加载 Author.csv 数据
    Paper = pandas.read_csv(config.PAPER_FILE)              # 加载 Paper.csv 数据
    Journal = pandas.read_csv(config.JOURNAL_FILE)          # 加载 Journal.csv 数据
    Conference = pandas.read_csv(config.CONFERENCE_FILE)    # 加载 Conference.csv 数据
    print("data is loaded...")

    # 为训练和测试数据，抽取特征，分别生成特征文件
    #trainer.make_feature_file(train_AuthorIdPaperIds, test_AuthorIdPaperIds, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference)
    
    for i in range(1):
        # 根据训练特征文件，训练模型
        trainer.train_mode()
        # 使用训练好的模型，对测试集进行预测
        trainer.test_model()
        # 对模型的预测结果，重新进行整理，得到想要的格式的预测结果
        get_prediction(config.TEST_FEATURE_PATH, config.TEST_RESULT_PATH, config.TEST_PREDICT_PATH)
        #break
        ''' 评估,（预测 vs 标准答案）'''
        gold_file = config.GOLD_FILE
        pred_file = config.TEST_PREDICT_PATH
        print("Test ",i)
        cmd = "python evalution.py %s %s" % (gold_file, pred_file)
        os.system(cmd)


