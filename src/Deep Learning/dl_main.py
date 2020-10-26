import FCN
import utils

import pandas as pd
import numpy as np
import dl_settings as settings


if __name__ == "__main__":
    df = pd.read_csv(settings.INPUT_PATH_MCYT_GENUINE)
    x_train, y_train, x_test, y_test = utils.split_data(df)

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    y_true = np.argmax(y_test, axis=0)

    shape = x_train.shape[1:]

    classifier = FCN.Classifier_FCN(
        output_directory=settings.OUTPUT_PATH, input_shape=shape, nb_classes=18)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)

    print(classifier.predict(x_test, y_true, x_train, y_train, y_test))