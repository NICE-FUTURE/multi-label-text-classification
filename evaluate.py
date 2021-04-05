# -*- "coding: utf-8" -*-

import pickle
import sys

import numpy as np
from keras.models import load_model

from config import Config
from data import DataLoader

model_path = "best_model.h5"

def evaluate(model=None):
    with open("./data/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open("./data/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(Config.val_path, "r", encoding="utf-8") as f:
        n_val = 0
        for line in f:
            n_val += 1

    n_labels = len(labels)
    test_loader = DataLoader(Config.val_path, batch_size=Config.batch_size, n_labels=n_labels, vectorizer=vectorizer)

    if model is None:
        model = load_model(model_path)

    y_predict = model.predict_generator(generator=test_loader.get_test_generator(), steps=n_val//Config.batch_size+1, verbose=0)

    with open("./prediction.csv", "w", encoding="utf-8-sig") as fout:
        with open(Config.val_path, "r", encoding="utf-8") as fin:
            cnt = 0
            hit = 0
            gt_total = 0
            pt_total = 0
            for line in fin:
                cur_labels, description = line.strip().split(",")
                ground_truth = [labels[idx] for idx in map(int, cur_labels.split())]
                predict = [labels[idx] for idx in np.where(y_predict[cnt]>0.5)[0]]

                cur_hit = len(set(ground_truth).intersection(set(predict)))
                hit += cur_hit
                gt_total += len(ground_truth)
                pt_total += len(predict)

                fout.write("{},{},{}\n".format(" ".join(ground_truth), " ".join(predict), cur_hit))
                cnt += 1

    precision = hit/pt_total if pt_total != 0 else 0
    recall = hit/gt_total if gt_total != 0 else 0
    print("precision:{:.4f}, recall:{:.4f}".format(precision, recall))
    return precision, recall


def evaluate_trainset(model=None):
    with open("./data/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open("./data/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(Config.train_path, "r", encoding="utf-8") as f:
        n_val = 0
        for line in f:
            n_val += 1

    n_labels = len(labels)
    test_loader = DataLoader(Config.train_path, batch_size=Config.batch_size, n_labels=n_labels, vectorizer=vectorizer)

    if model is None:
        model = load_model(model_path)

    y_predict = model.predict_generator(generator=test_loader.get_test_generator(), steps=n_val//Config.batch_size+1, verbose=0)

    with open("./prediction_trainset.csv", "w", encoding="utf-8-sig") as fout:
        with open(Config.train_path, "r", encoding="utf-8") as fin:
            cnt = 0
            hit = 0
            gt_total = 0
            pt_total = 0
            for line in fin:
                cur_labels, description = line.strip().split(",")
                ground_truth = [labels[idx] for idx in map(int, cur_labels.split())]
                predict = [labels[idx] for idx in np.where(y_predict[cnt]>0.5)[0]]

                cur_hit = len(set(ground_truth).intersection(set(predict)))
                hit += cur_hit
                gt_total += len(ground_truth)
                pt_total += len(predict)

                fout.write("{},{},{}\n".format(" ".join(ground_truth), " ".join(predict), cur_hit))
                cnt += 1

    precision = hit/pt_total if pt_total != 0 else 0
    recall = hit/gt_total if gt_total != 0 else 0
    print("precision:{:.4f}, recall:{:.4f}".format(precision, recall))
    return precision, recall


def add_description(n_train, n_val):
    """将原始岗位描述加入到预测结果中
    """
    with open("./prediction_with_description.csv", "w", encoding="utf-8-sig") as fout:
        fout.write("岗位描述,真实标签,预测标签,命中数量\n")
        with open("./data/label_description.csv", "r", encoding="utf-8") as fin1:
            with open("./prediction.csv", "r", encoding="utf-8") as fin2:
                for i in range(n_train):
                    line = fin1.readline()
                for i in range(n_val):
                    line = fin1.readline()
                    labels, description = line.strip().split(",")
                    fout.write(description + "," + fin2.readline())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    evaluate_trainset()
    evaluate()
    add_description(n_train=90000, n_val=10000)
