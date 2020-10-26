import settings
import resampling
import classification

import pandas as pd

if __name__ == '__main__':
    # resampling.resample_all()

    df_mcyt_genuine = pd.read_csv(settings.MCYT_GENUINE)
    df_mcyt_forgery = pd.read_csv(settings.MCYT_FORGERY)

    # 10 fold cross validation
    classification.evaluate_cross_validation(df_mcyt_genuine)

    # train(genuine) - test(forgery) evaluation
    classification.evaluate_train_test(df_mcyt_genuine, df_mcyt_forgery)
