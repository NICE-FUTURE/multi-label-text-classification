# -*- "coding: utf-8" -*-

class Config(object):

    train_path = "./data/train.csv"
    val_path = "./data/val.csv"
    test_path = ""
    max_features = 800  # 重新执行 prepare_data.py
    batch_size = 16
    epochs = 100
    n_train = 20
    n_val = 10
