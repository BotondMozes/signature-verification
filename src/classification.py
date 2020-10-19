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


def evaluate_cross_validation(df, num_folds=10):
    x_train, y_train = get_data(df)

    forest_model = RandomForestClassifier(n_jobs=-1)
    forest_model.fit(x_train, y_train.values.ravel())

    score = np.mean(cross_val_score(forest_model, x_train,
                                    y_train.values.ravel(), cv=num_folds, n_jobs=-1))

    print(str(num_folds) + " fold cross validation score: " + str(score))


def evaluate_train_test(df_train, df_test):
    x_train, y_train = get_data(df_train)

    forest_model = RandomForestClassifier(n_jobs=-1)
    forest_model.fit(x_train, y_train.values.ravel())

    x_test, y_test = get_data(df_test)

    y_pred_test = forest_model.predict(x_test)

    score = accuracy_score(y_test, y_pred_test)

    print("Evaluation score: " + str(score))
