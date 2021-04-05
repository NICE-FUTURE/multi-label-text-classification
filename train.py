# -*- "coding: utf-8" -*-

import pickle
import time
import os

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import matplotlib.pyplot as plt

from models import *
from data import DataLoader
from config import Config
from evaluate import *


def train(continuous=False):

    with open("./data/labels.pkl", "rb") as f:
        n_labels = len(pickle.load(f))
    with open("./data/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(Config.train_path, "r", encoding="utf-8") as f:
        n_train = 0
        for line in f:
            n_train += 1
    with open(Config.val_path, "r", encoding="utf-8") as f:
        n_val = 0
        for line in f:
            n_val += 1

    batch_size = Config.batch_size
    epochs = Config.epochs

    # 加载数据
    train_loader = DataLoader(Config.train_path, batch_size=batch_size, n_labels=n_labels, vectorizer=vectorizer)

    # 设定模型
    if continuous:
        model = load_model("./best_model.h5")
    else:
        # model = Net(Config.max_features, n_labels)
        model = Net1(Config.max_features, n_labels)
    model.summary()

    # 训练模型
    start_time = time.time()
    checkpoint = ModelCheckpoint("./best_model.h5", monitor='loss', verbose=0, 
            save_best_only=True, mode='auto', period=1)  # 监控并保存最低loss的模型
    history = {"loss":[], "precision":[], "recall":[], "val_precision":[], "val_recall":[]}
    for epoch in range(epochs):
        print("\n[Epoch:", epoch, "]")
        cur_history = model.fit_generator(
                generator=train_loader.get_train_generator(), steps_per_epoch=n_train//batch_size+1,
                epochs=1, verbose=2, 
                callbacks=[checkpoint]
            )
        history["loss"].append(cur_history.history["loss"][0])
        print("evaluate trainset:")
        precision, recall = evaluate_trainset(model)
        history["precision"].append(precision)
        history["recall"].append(recall)
        print("evaluate testset:")
        val_precision, val_recall = evaluate(model)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
    stop_time = time.time()
    print("\n[训练耗时: {:.2f}分钟]\n".format((stop_time-start_time)/60))

    # 可视化训练过程
    plt.figure(figsize=(10,8))
    plt.subplot(111)
    plt.plot(history["loss"], color="blue", label="loss")
    plt.legend()
    plt.savefig("./history/history-loss-{}.png".format(time.time()), dpi=100)

    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.plot(history["precision"], color="blue", label="precision")
    plt.plot(history["val_precision"], color="orange", label="val_precision")
    plt.legend()
    plt.subplot(212)
    plt.plot(history["recall"], color="blue", label="recall")
    plt.plot(history["val_recall"], color="orange", label="val_recall")
    plt.legend()
    plt.savefig("./history/history-evaluate-{}.png".format(time.time()), dpi=100)


if __name__ == "__main__":
    if not os.path.exists("./history/"):
        os.mkdir("./history/")
    train(continuous=False)
    # train(continuous=True)
