#coding:utf-8
import os, config
from sklearn.datasets import load_svmlight_file
from sklearn import svm, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#抽象类 所有的分类器都以此类为父类进行派生
class Strategy(object):
    def train_model(self, train_file_path, model_path):
        return None
    def test_model(self, test_file_path, model_path, result_file_path):
        return None

#主分类器 用于调用各个分类器
class Classifier(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def train_model(self, train_file_path, model_path):
        self.strategy.train_model(train_file_path, model_path)

    def test_model(self, test_file_path, model_path, result_file_path):
        self.strategy.test_model(test_file_path, model_path, result_file_path)



''' skLearn '''
'''
每个模型都有两个类成员和两个成员函数
trainer : 分类器名称,在主函数运行时用于打印提示字符
clf     : 即classifier,分类器对象
train_model : 训练模型函数 
test_model:   测试模型函数
所有分类器的函数流程大同小异
在这里做统一解释:
train_model:
(1)通过load_svmlight_file加载特征文件
(2)调用sklearn的库进行模型训练

test_model:
(1)通过load_svmlight_file加载特征文件
(2)调用sklearn的库进行模型预测
(3)将预测结果写入结果文件当中
'''
#模型1: 决策树
class skLearn_DecisionTree(Strategy):
    def __init__(self):
        self.trainer = "skLearn decisionTree"
        self.clf = tree.DecisionTreeClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))

#模型2: 朴素贝叶斯
class skLearn_NaiveBayes(Strategy):
    def __init__(self):
        self.trainer = "skLearn NaiveBayes"
        self.clf = GaussianNB()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        train_X = train_X.toarray()
        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        test_X = test_X.toarray()
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))

#模型3: 支持向量机
class skLearn_svm(Strategy):
    def __init__(self):
        self.trainer = "skLearn svm"
        self.clf = svm.LinearSVC()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))

#模型4: 逻辑回归
class skLearn_lr(Strategy):
    def __init__(self):
        self.trainer = "skLearn LogisticRegression"
        self.clf = LogisticRegression()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))

#模型5: k近邻算法(default k = 3)
class skLearn_KNN(Strategy):
    def __init__(self):
        self.trainer = "skLearn KNN"
        self.clf = KNeighborsClassifier(n_neighbors=10)
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


#模型6: 集成学习
class skLearn_AdaBoostClassifier(Strategy):
    def __init__(self):
        self.trainer = "skLearn AdaBoostClassifier"
        self.clf = AdaBoostClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))

#模型7: 随机森林分类
class sklearn_RandomForestClassifier(Strategy):
    def __init__(self):
        self.trainer = "skLearn RandomForestClassifier"
        self.clf = RandomForestClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))

#模型8: 投票分类(hard—少数服从多数)
class sklearn_VotingClassifier(Strategy):
    def __init__(self):
        self.trainer = "skLearn VotingClassifier"

        clf1 = tree.DecisionTreeClassifier()
        clf2 = GaussianNB()
        clf3 = LogisticRegression()
        clf4 = svm.LinearSVC()
        clf5 = KNeighborsClassifier(n_neighbors=3)
        clf6 = AdaBoostClassifier()
        clf7 = RandomForestClassifier()
        self.clf = VotingClassifier(estimators=[('dtc',clf1), ('lr', clf3), ('svm', clf4), ('knc',clf5), ('ada', clf6), ('rfc',clf7)], voting='hard')

        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


if __name__ == "__main__":
    pass
