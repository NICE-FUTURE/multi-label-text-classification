# -*- "coding: utf-8" -*-

import os
import sys
import random
import pickle
import time

import numpy as np
from keras.preprocessing.sequence import pad_sequences


class DataLoader(object):

    def __init__(self, filepath, batch_size, n_labels, vectorizer):
        self.filepath = filepath
        self.batch_size = batch_size
        self.n_labels = n_labels
        self.vectorizer = vectorizer

    def get_train_generator(self):
        while True:
            x_trains = None
            y_trains = None
            count = 0
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    x_train, y_train = self.convert_line(line)
                    x_trains = x_train if x_trains is None else np.append(x_trains, x_train, axis=0)
                    y_trains = y_train if y_trains is None else np.append(y_trains, y_train, axis=0)
                    count += 1
                    if count == self.batch_size:
                        yield x_trains, y_trains
                        count = 0
                        x_trains = None
                        y_trains = None
                if count > 0:
                    yield x_trains, y_trains
                    count = 0
                    x_trains = None
                    y_trains = None

    def get_test_generator(self):
        while True:
            x_trains = None
            count = 0
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    x_train, _ = self.convert_line(line)
                    x_trains = x_train if x_trains is None else np.append(x_trains, x_train, axis=0)
                    count += 1
                    if count == self.batch_size:
                        yield x_trains
                        count = 0
                        x_trains = None
                if count > 0:
                    yield x_trains
                    count = 0
                    x_trains = None

    def convert_line(self, line):
        labels, description = line.strip().split(",")
        # 标签使用 one hot 编码
        indices = map(int, labels.split())
        y_sample = np.zeros((1,self.n_labels))
        for idx in indices:
            y_sample[0][idx] = 1
        # 描述使用词汇索引编码
        x_sample = self.vectorizer.transform([description])
        x_sample = x_sample.toarray()
        return x_sample, y_sample


if __name__ == "__main__":
    '''
    测试当前类是否有效
    '''
    from config import Config
    with open("./data/labels.pkl", "rb") as f:
        n_labels = len(pickle.load(f))
    with open("./data/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    data = DataLoader("./data/train.csv", batch_size=20, n_labels=n_labels, vectorizer=vectorizer)
    train_generator = data.get_train_generator()
    for i in range(2):
        (x_trains, y_trains) = train_generator.__next__()
        print(x_trains.shape)
        print(y_trains.shape)
    test_generator = data.get_test_generator()
    for i in range(2):
        x_tests = test_generator.__next__()
        print(x_tests.shape)
