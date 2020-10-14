import settings
import os
import csv

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def speed(df):
    result = df.diff()
    result.columns = ["x1", "y1", "p1"]
    result["x1"][0] = 0
    result["y1"][0] = 0
    result["p1"][0] = 0

    return result


def resample_mcyt():
    genuine_result = []
    forgery_result = []

    folder_list = listdirs(settings.MCYT_INTERP)

    for folder in folder_list:
        signature_list = os.listdir(settings.MCYT_INTERP + folder)

        if len(signature_list) != 0:
            for file1 in signature_list:
                df = pd.read_csv(settings.MCYT_INTERP +
                                 folder + "/" + file1, usecols=[1, 2, 3])

                df = speed(df)
                scaler = MinMaxScaler()

                df = scaler.fit_transform(df)

                res = []
                res = np.append(res, df[:, 0])
                res = np.append(res, df[:, 1])
                res = np.append(res, df[:, 2])
                res = res.tolist()
                res.append(folder)

                if file1[4] == 'v':
                    genuine_result.append(res)
                else:
                    forgery_result.append(res)

    with open(settings.FINAL_OUTPUT+"mcyt_genuine.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(genuine_result)

    with open(settings.FINAL_OUTPUT+"mcyt_forgery.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(forgery_result)


def resample_mobisig():
    genuine_result = []
    forgery_result = []

    folder_list = listdirs(settings.MOBISIG_INTERP)

    for folder in folder_list:
        signature_list = os.listdir(settings.MOBISIG_INTERP + folder)

        if len(signature_list) != 0:
            for file1 in signature_list:
                df = pd.read_csv(settings.MOBISIG_INTERP +
                                 folder + "/" + file1, usecols=[1, 2, 3])

                df = speed(df)
                scaler = MinMaxScaler()

                df = scaler.fit_transform(df)

                res = []
                res = np.append(res, df[:, 0])
                res = np.append(res, df[:, 1])
                res = np.append(res, df[:, 2])
                res = res.tolist()
                res.append(folder)

                if "GEN" in file1:
                    genuine_result.append(res)
                else:
                    forgery_result.append(res)

    with open(settings.FINAL_OUTPUT+"mobisig_genuine.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(genuine_result)

    with open(settings.FINAL_OUTPUT+"mobisig_forgery.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(forgery_result)


def resample_all():
    resample_mcyt()
    resample_mobisig()


resample_all()
