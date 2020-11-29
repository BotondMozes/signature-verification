from RandomForest.classification import evaluate_cross_validation
import DeepLearning.FCN as FCN
import DeepLearning.dl_settings as settings
from DeepLearning.utils import split_data
import DeepLearning.dl_main as dl
import pandas as pd


def evaluate_all():
    df = pd.read_csv(settings.INPUT_PATH_MCYT_GENUINE, header=None)
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(df)
    dataset = "MCYT"
    columns = ["DATASET", "MODEL", "PARAMETER", "SCORE"]
    result = []

    for trees in [4, 8, 16, 32, 64, 128, 256]:
        score = evaluate_cross_validation(
            trees, x_train, y_train, x_test, y_test, 10)
        result.append([dataset, "RandomForest(10 fold CV)", trees, score])

    result = dl.test_filters(dataset, result, x_train, y_train,
                             x_test, y_test, x_val, y_val)

    df = pd.read_csv(settings.INPUT_PATH_MOBISIG_GENUINE, header=None)
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(df)
    dataset = "MOBISIG"

    for trees in [4, 8, 16, 32, 64, 128, 256]:
        score = evaluate_cross_validation(
            trees, x_train, y_train, x_test, y_test, 10)
        result.append([dataset, "RandomForest(10 fold CV)", trees, score])

    result = dl.test_filters(dataset, result, x_train, y_train,
                             x_test, y_test, x_val, y_val)

    df = pd.DataFrame(data=result, columns=columns)
    df.to_csv("./evaluation.csv")


if __name__ == '__main__':
    # resampling.resample_all()

    # df_mcyt_genuine = pd.read_csv(settings.MCYT_GENUINE)
    # df_mcyt_forgery = pd.read_csv(settings.MCYT_FORGERY)

    # 10 fold cross validation
    # classification.evaluate_cross_validation(df_mcyt_genuine)

    # train(genuine) - test(forgery) evaluation
    # classification.evaluate_train_test(df_mcyt_genuine, df_mcyt_forgery)

    evaluate_all()
