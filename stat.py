import os
import csv
import settings

import pandas as pd
import matplotlib.pyplot as plt


def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def get_stat(path, data_type, output):
    result = [["type", "length"]]
    folder_list = listdirs(path)

    for folder in folder_list:
        signature_list = os.listdir(path + folder)
        if len(signature_list) != 0:
            for file1 in signature_list:
                df = pd.read_csv(path + folder + "/" + file1)
                result.append([data_type + "/" + folder +
                               "/" + file1, len(df.index)])

    with open(output, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)


def make_stat():
    get_stat(settings.MCYT_DATA, "MCYT", settings.MCYT_STAT_DATA)
    get_stat(settings.MOBISIG_DATA, "MOBISIG", settings.MOBISIG_STAT_DATA)


def get_mean_length(path):
    df = pd.read_csv(path, usecols=[1])

    sum_length = df.sum(axis=0)
    mean = sum_length / len(df.index)

    print(mean)


def plot_histogram(path):
    df = pd.read_csv(path)
    df.hist(bins=250)
    plt.show()


if __name__ == '__main__':
    # make_stat()
    get_mean_length(settings.MCYT_STAT_DATA)
    get_mean_length(settings.MOBISIG_STAT_DATA)

    plot_histogram(settings.MCYT_STAT_DATA)
    plot_histogram(settings.MOBISIG_STAT_DATA)
