import settings

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def get_data(df):
    y = df.iloc[:, -1:]
    x = df.iloc[:, 0:1536]

    return x, y


def evaluate_cross_validation(num_trees, x_train, y_train, x_test, y_test, num_folds=10):
    x_train = x_train.reshape(-1, 1536)
    x_test = x_test.reshape(-1, 1536)

    y_train = np.where(y_train == 1)
    y_train = y_train[1]

    y_test = np.where(y_test == 1)
    y_test = y_test[1]
    forest_model = RandomForestClassifier(n_jobs=-1, n_estimators=num_trees)
    forest_model.fit(x_train, y_train)

    score = np.mean(cross_val_score(forest_model, x_test,
                                    y_test, cv=num_folds, n_jobs=-1))

    return score


def evaluate_train_test(df_train, df_test):
    x_train, y_train = get_data(df_train)

    forest_model = RandomForestClassifier(n_jobs=-1)
    forest_model.fit(x_train, y_train.values.ravel())

    x_test, y_test = get_data(df_test)

    y_pred_test = forest_model.predict(x_test)

    score = accuracy_score(y_test, y_pred_test)

    print("Evaluation score: " + str(score))
