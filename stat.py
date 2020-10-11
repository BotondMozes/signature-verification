import os
import csv
import settings

import pandas as pd
import matplotlib.pyplot as plt


def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def get_stat(path, result, data_type):
    folder_list = listdirs(path)

    for folder in folder_list:
        signature_list = os.listdir(path + folder)
        if len(signature_list) != 0:
            for file1 in signature_list:
                df = pd.read_csv(path + folder + "/" + file1)
                result.append([data_type + "/" + folder +
                               "/" + file1, len(df.index)])

    return result


def make_stat():
    result = [["type", "length"]]
    result = get_stat(settings.MCYT_DATA, result, "MCYT")
    result = get_stat(settings.MOBISIG_DATA, result, "MOBISIG")

    with open(settings.STAT_DATA, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)


def get_mean_length():
    df = pd.read_csv(settings.STAT_DATA, usecols=[1])

    sum_length = df.sum(axis=0)

    mean = sum_length / len(df.index)

    print(mean)


def plot_histogram():
    df = pd.read_csv(settings.STAT_DATA)
    df.hist(bins=250)
    plt.show()


if __name__ == '__main__':
    # make_stat()
    # get_mean_length()
    plot_histogram()
