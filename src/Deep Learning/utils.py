import dl_settings as stt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import FCN


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    # res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
    #                    columns=['precision', 'accuracy', 'recall', 'duration'])
    # res['precision'] = precision_score(y_true, y_pred, average='macro')
    # res['accuracy'] = accuracy_score(y_true, y_pred)

    # if not y_true_val is None:
    #     # this is useful when transfer learning is used with cross validation
    #     res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    # res['recall'] = recall_score(y_true, y_pred, average='macro')
    # res['duration'] = duration
    # return res
    return accuracy_score(y_true, y_pred)


def split_data(df):
    df = normalize_rows_signature(df, "ZSCORE")
    array = df.values

    nsamples, nfeatures = array.shape

    nfeatures = nfeatures - 1

    X = array[:, 0:nfeatures]
    y = array[:, -1]

    X = np.asarray(X).astype(np.float32)
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    enc = OneHotEncoder()
    enc.fit(y.reshape(-1, 1))
    y = enc.transform(y.reshape(-1, 1)).toarray()

    # train, test = train_test_split(X, y, test_size=0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=stt.RANDOM_STATE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=stt.RANDOM_STATE)

    return X_train, y_train, X_test, y_test, X_val, y_val


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


def test_filters(path, dtype, result):
    df = pd.read_csv(path)

    x_train, y_train, x_test, y_test, x_val, y_val = split_data(df)

    y_true = np.argmax(y_test, axis=1)

    nb_classes = y_test.shape[1]

    shape = x_train.shape[1:]

    for n in [4, 8, 16, 32, 64, 128]:
        classifier = FCN.Classifier_FCN(
            output_directory=stt.OUTPUT_PATH, input_shape=shape, nb_classes=nb_classes, verbose=True, num_filters=n)

        classifier.fit(x_train, y_train, x_val, y_val, y_true)

        accuracy = classifier.predict(x_test, y_true, x_train, y_train, y_test)

        result.loc[len(result)] = [dtype, n, accuracy]

    return result


def normalize_rows_signature(df, norm_type):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]

    rows, cols = X.shape
    if(norm_type == "MINMAX"):
        for i in range(0, rows):
            row = X[i, :]
            maxr = max(row)
            minr = min(row)
            if(maxr != minr):
                X[i, :] = (X[i, :] - minr) / (maxr - minr)
            else:
                X[i, :] = 1
    if(norm_type == "ZSCORE"):
        for i in range(0, rows):
            row = X[i, :]
            rows = np.array_split(row, 3)
            row1 = rows[0]
            row2 = rows[1]
            row3 = rows[2]
            mu1 = np.mean(row1)
            sigma1 = np.std(row1)

            mu1 = np.mean(row1)
            sigma1 = np.std(row1)

            mu2 = np.mean(row2)
            sigma2 = np.std(row2)

            mu3 = np.mean(row3)
            sigma3 = np.std(row3)

            row1 = (row1 - mu1) / sigma1
            row2 = (row2 - mu2) / sigma2
            row3 = (row3 - mu3) / sigma3

            X[i, :] = np.concatenate((row1, row2, row3), axis=0)

    df = pd.DataFrame(X)
    df['user'] = y
    return df
