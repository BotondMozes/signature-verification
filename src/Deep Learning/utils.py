import dl_settings as stt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def split_data(df):
    array = df.values

    nsamples, nfeatures = array.shape

    nfeatures = nfeatures - 1

    X = array[:, 0:nfeatures]
    y = array[:, -1]

    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    enc = OneHotEncoder()
    enc.fit(y.reshape(-1, 1))
    y = enc.transform(y.reshape(-1, 1)).toarray()

    # train, test = train_test_split(df, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=stt.RANDOM_STATE)

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train, y_train, test_size=0.25, random_state=stt.RANDOM_STATE)
    return X_train, y_train, X_test, y_test


def plot_training(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
