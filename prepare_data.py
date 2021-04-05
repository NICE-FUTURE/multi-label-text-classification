# -*- "coding: utf-8" -*-

import os
import pickle

from config import Config


def clean_description():
    import jieba

    with open("./data/stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = f.read().split("\n")[:-1]
    with open("./data/label_description.csv", "r", encoding="utf-8") as fin:
        with open("./data/label_description_clean.csv", "w", encoding="utf-8") as fout:
            for line in fin:
                label, description = line.strip().split(",")
                # 分词 + 去除停用词
                words_str = " ".join([word for word in jieba.cut(description) if word not in stopwords])
                fout.write("{},{}\n".format(label, words_str))


def label_to_index():
    """对label进行编码
    """
    # 找出所有label
    labels_set = set()
    with open("./data/label_description_clean.csv", "r", encoding="utf-8") as f:
        for line in f:
            labels = line.strip().split(",")[0].split()
            for label in labels:
                labels_set.add(label)
    # 构建label:index字典
    labels = list(labels_set)
    label2index = {label:idx for idx,label in enumerate(labels)}
    with open("./data/labels.pkl", "wb") as f:
        pickle.dump(labels, f, -1)
    with open("./data/label2index.pkl", "wb") as f:
        pickle.dump(label2index, f, -1)
    # 将数据中的label替换成index
    with open("./data/label_description_clean.csv", "r", encoding="utf-8") as fin:
        with open("./data/label_encode_description.csv", "w", encoding="utf-8") as fout:
            for line in fin:
                labels, description = line.strip().split(",")
                labels = labels.split()
                label_indices = [str(label2index[label]) for label in labels]
                fout.write("{},{}\n".format(" ".join(label_indices), description))


def split_train_val(n_train, n_val):
    ftrain = open("./data/train.csv", "w", encoding="utf-8")
    fval = open("./data/val.csv", "w", encoding="utf-8")
    with open("./data/label_encode_description.csv", "r", encoding="utf-8") as fin:
        for i in range(n_train):
            line = fin.readline()
            ftrain.write(line)
        for i in range(n_val):
            line = fin.readline()
            fval.write(line)
    ftrain.close()
    fval.close()


def train_description_tokenizer():
    from keras.preprocessing.text import Tokenizer

    def text_iterator():
        with open("./data/label_encode_description.csv", "r", encoding="utf-8") as f:
            for line in f:
                labels, description = line.strip().split(",")
                yield description

    tokenizer = Tokenizer(num_words=Config.vocabulary_max_len)
    tokenizer.fit_on_texts(text_iterator())
    with open("./data/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f, -1)


def train_description_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    def text_iterator():
        with open("./data/label_encode_description.csv", "r", encoding="utf-8") as f:
            for line in f:
                labels, description = line.strip().split(",")
                yield description
    
    vectorizer = TfidfVectorizer(max_df=0.7, max_features=Config.max_features)  # max_df 可过滤掉常用的无意义词
    vectorizer.fit(text_iterator())
    with open("./data/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f, -1)


if __name__ == "__main__":
    clean_description()
    label_to_index()
    split_train_val(n_train=Config.n_train, n_val=Config.n_val)
    train_description_vectorizer()
