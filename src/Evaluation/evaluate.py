from RandomForest.classification import evaluate_cross_validation


def evaluate_all():
    df = pd.read_csv(settings.INPUT_PATH_MCYT_GENUINE, header=None)
    x_train, y_train, x_test, y_test, x_val, y_val = utils.split_data(df)

    for trees in [4, 8, 16, 32, 64, 128]:
        print("asd")


if __name__ == "__main__":
    evaluate_all()
