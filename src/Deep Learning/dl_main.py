import FCN
import utils

import pandas as pd
import numpy as np
import dl_settings as settings


if __name__ == "__main__":
    df = pd.read_csv(settings.INPUT_PATH_MCYT_GENUINE, header=None)

    print(df.shape)

    x_train, y_train, x_test, y_test, x_val, y_val = utils.split_data(df)

    y_true = np.argmax(y_test, axis=1)

    nb_classes = y_test.shape[1]

    shape = x_train.shape[1:]

    # classifier = FCN.Classifier_FCN(
    #     output_directory=settings.OUTPUT_PATH, input_shape=shape, nb_classes=nb_classes, verbose=True, num_filters=128)

    # classifier.fit(x_train, y_train, x_val, y_val, y_true)

    # accuracy = classifier.predict(x_test, y_true, x_train, y_train, y_test)

    res = pd.DataFrame(columns=['Database', 'num_filters', 'Accuracy'])
    res = utils.test_filters(settings.INPUT_PATH_MCYT_GENUINE, 'MCYT', res)
    res = utils.test_filters(
        settings.INPUT_PATH_MOBISIG_GENUINE, 'MOBISIG', res)

    # print(accuracy)
    print(res)
