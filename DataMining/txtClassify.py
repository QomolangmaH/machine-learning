import numpy as np
import pandas as pd
import re
import jieba
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
# from gensim.models import Word2Vec

import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict

K = 50  # 定义选择的特征数量K


# %% 读取数据
def data_preprocess():
    data_raw = pd.read_csv('waimai_10k.csv')
    # data_raw = pd.read_csv('shui guo.csv')
    # data_raw.head()
    # data_raw.info()
    data = data_raw['review']
    data.head()

    # 数据清洗
    data = data.fillna('')  # 用空字符串填充缺失值
    data = data.apply(lambda x: x.strip())  # 去除文本开头和结尾的空白字符
    data = data.apply(lambda x: x.replace('\n', ' '))  # 将换行符替换为空格
    data = data.apply(lambda x: re.sub('[0-9]', '', x))  # 去除数字
    data = data.apply(lambda x: re.sub("[^a-zA-Z\u4e00-\u9fff]", ' ', x))  # 去除非汉字和字母的非空白字符

    # 文本预处理
    data = data.apply(lambda x: ' '.join(jieba.cut(x)))  # 使用jieba分词
    # 停用词列表
    stopwords = ["吧", "是", "的", "了", "啦", "得", "么", "在", "并且", "因此", "因为", "所以", "虽然", "但是"]
    data = data.apply(lambda x: ' '.join([i for i in x.split() if i not in stopwords]))  # 去停用词
    # 移除低频词
    word_counts = Counter(' '.join(+data).split())
    low_freq_words = [word for word, count in word_counts.items() if count < 3]
    data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in low_freq_words]))

    return data_raw, data


# %%
def calculate_tfidf(corpus):
    # 分词
    tokenized_corpus = [jieba.lcut(text) for text in corpus]
    # 构建词频统计
    word_counts = []
    for tokens in tokenized_corpus:
        word_count = {}
        for token in tokens:
            if token not in word_count:
                word_count[token] = 0
            word_count[token] += 1
        word_counts.append(word_count)
    # 计算逆文档频率（IDF）
    idf = {}
    num_documents = len(corpus)
    for word_count in word_counts:
        for word in word_count:
            if word not in idf:
                idf[word] = 0
            idf[word] += 1

    for word in idf:
        idf[word] = math.log(
            num_documents / (idf[word] + 1))

    # 计算 TF-IDF
    tf_idf_vectors = []
    for word_count in word_counts:
        tf_idf_vector = {}
        for word in word_count:
            tf_idf_vector[word] = word_count[word] * idf[word]
        tf_idf_vectors.append(tf_idf_vector)

    return tf_idf_vectors


def text_classification_1():
    X = []
    tf_idf_vectors = calculate_tfidf(Data)

    # 构建词汇表
    vocabulary = set()
    for tf_idf_vector in tf_idf_vectors:
        vocabulary.update(tf_idf_vector.keys())
    for tf_idf_vector in tf_idf_vectors:
        features = [tf_idf_vector.get(word, 0) for word in vocabulary]
        X.append(features)
    Metric = []
    for j in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, Data_raw['label'],
                                                            shuffle=True, test_size=0.3, random_state=j)
        methods = OrderedDict()
        # methods["knn"] = KNeighborsClassifier()
        # methods["svm"] = svm.SVC()
        methods["mlp"] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
        # # clf = GaussianNB()          # 高斯朴素贝叶斯
        # # clf = MultinomialNB()       # 多项分布朴素贝叶斯
        # # clf = BernoulliNB()  # 伯努利朴素贝叶斯
        # methods["nb"] = BernoulliNB()
        # methods["dt"] = DecisionTreeClassifier()
        # methods["rf"] = RandomForestClassifier(n_estimators=100)
        metric = []
        for i, (label, method) in enumerate(methods.items()):
            classifier = method
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
            confusion = metrics.confusion_matrix(y_test, y_pred)
            metric.append([accuracy, precision, recall, f1])
            print(label, accuracy, precision, recall, f1)
            print(confusion)
        Metric += metric
    return Metric


def tfidf_train(data, label):
    # TF-IDF特征提取
    # https://blog.csdn.net/blmoistawinde/article/details/80816179
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.2, min_df=0.0001)
    data_tfidf = vec.fit_transform(data)
    print(vec.get_feature_names())
    print(data_tfidf.shape)
    # 保存特征提取器
    with open('tfidf.pkl', 'wb') as file:
        pickle.dump(vec, file)
    # 特征选择
    selector = SelectKBest(chi2, k=K)
    data_selected = selector.fit_transform(data_tfidf, label)
    return data_selected


def tfidf_test(data, label):
    # 加载特征提取器
    with open('tfidf.pkl', 'rb') as file:
        vec = pickle.load(file)

    test = vec.transform(data)
    # 特征选择
    selector = SelectKBest(chi2, k=K)
    test_selected = selector.fit_transform(test, label)
    return test_selected


def text_classification_2():
    Metric = []
    for j in range(20):
        X_train, X_test, y_train, y_test = train_test_split(Data, Data_raw['label'],
                                                            shuffle=True, test_size=0.3, random_state=j)
        methods = OrderedDict()
        methods["knn"] = KNeighborsClassifier()
        methods["svm"] = svm.SVC()
        methods["mlp"] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
        # clf = GaussianNB()          # 高斯朴素贝叶斯
        # clf = MultinomialNB()       # 多项分布朴素贝叶斯
        # clf = BernoulliNB()         # 伯努利朴素贝叶斯
        methods["nb"] = BernoulliNB()
        methods["dt"] = DecisionTreeClassifier()
        methods["rf"] = RandomForestClassifier(n_estimators=100)
        train = tfidf_train(X_train, y_train)
        metric = []
        for i, (label, method) in enumerate(methods.items()):
            clf = method
            clf.fit(train, y_train)
            test = tfidf_test(X_test, y_test)
            y_pred = clf.predict(test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
            confusion = metrics.confusion_matrix(y_test, y_pred)
            metric.append([accuracy, precision, recall, f1])
            print(label, accuracy, precision, recall, f1)
            # print(confusion)
        Metric += metric
    return Metric


def svm_grid(X_train, y_train):
    param_grid = [{
        # 'C':
        'kernel': ['linear',  # 线性核函数
                   'poly',  # 多项式核函数
                   'rbf',  # 高斯核
                   'sigmoid'  # sigmod核函数
                   # 'precomputed'    # 核矩阵
                   ],  # 核函数类型，
        'degree': np.arange(2, 5, 1),  # int, 多项式核函数的阶数， 这个参数只对多项式核函数有用,默认为3
        # 'gamma': np.arange(1e-6, 1e-4, 1e-5)   # float, 核函数系数,只对’rbf’ ,’poly’ ,’sigmod’有效, 默认为样本特征数的倒数，即1/n_features。
        # 'coef0' # float，核函数中的独立项, 只有对’poly’ 和,’sigmod’核函数有用, 是指其中的参数c。默认为0.0
    }]
    svc = svm.SVC(kernel='poly')
    # 网格搜索
    grid_search = GridSearchCV(svc,
                               param_grid,
                               cv=10,
                               scoring="accuracy",
                               return_train_score=True)
    grid_search.fit(X_train, y_train)
    # 最优模型参
    final_model = grid_search.best_estimator_
    return final_model


# %%
if __name__ == '__main__':
    Data_raw, Data = data_preprocess()
    # # 手写TF-IDF
    metrics_ = text_classification_1()

    # 调包TF-IDF
    # metrics_ = text_classification_2()

    metrics_ = np.array(metrics_)
    for i in range(4):  # 输出各项指标平均值
        print(np.mean(metrics_[:, i]))
