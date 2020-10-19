import settings

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score


def get_data(path):
    df = pd.read_csv(path, skiprows=1, header=None)

    y = df.iloc[:, -1:]
    x = df.iloc[:, 0:1536]

    return x, y


if __name__ == '__main__':
    x_train, y_train = get_data(settings.MCYT_GENUINE)

    # for i in range(1, 11):
    #     forest_model = RandomForestRegressor(
    #         n_jobs=-1, random_state=1, max_leaf_nodes=33, max_depth=i)
    #     forest_model.fit(x_train, y_train.values.ravel())

    #     result = -1 * np.mean(cross_val_score(forest_model, x_train,
    #                                           y_train.values.ravel(), cv=10))

    #     print(str(result) + "-" + str(i))

    # forest_model = RandomForestClassifier(
    #     n_jobs=-1, random_state=1, n_estimators=800, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=100, bootstrap=False)

    forest_model = RandomForestClassifier(n_jobs=-1, random_state=1)
    forest_model.fit(x_train, y_train.values.ravel())

    result = np.mean(cross_val_score(forest_model, x_train,
                                     y_train.values.ravel(), cv=10))
    val_predict = forest_model.predict(x_train)
    # print(result)
    # print(val_predict)

    x_predict, y_predict = get_data(settings.MCYT_FORGERY)

    val_predict = forest_model.predict(x_predict)
    print(val_predict)
    print(y_predict)

    # {'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': False}
    # print(val_predict)
    # print(y_train)

    # rf = RandomForestRegressor()

    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}

    # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
    #                                n_iter=100, cv=10, verbose=2, random_state=42, n_jobs=-1)

    # rf_random.fit(x_train, y_train.values.ravel())

    # print(rf_random.best_params_)
