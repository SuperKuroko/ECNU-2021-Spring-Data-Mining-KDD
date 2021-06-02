#!/usr/bin/env python
#encoding: utf-8
import json
import sys
import importlib
importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("../")
import joblib
#from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn import svm

if __name__ == '__main__':
    # load_svmlight_file加载特征文件
    # 其中train_X为矩阵的格式
    # (i,j) val表示第i行第j个特征值为val
    # 而train_y为数组格式, train_y[i] = label表示第i行的标记值为label
    train_X, train_y = load_svmlight_file("feature/train.feature")
    test_X, test_y = load_svmlight_file("feature/test.feature")

    # SVC:Support Vector Classification 支持向量机的分类器
    clf = svm.SVC()

    # fit(input,result) 将train_X作为输入数据,train_y作为标准输出进行训练
    clf.fit(train_X, train_y)

    # 再以test_X为输入,输出以上述模型为依据的预测结果
    print(clf.predict(test_X))


