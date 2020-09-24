import os
import time

import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.preprocessing import MinMaxScaler


def dtw(filename1, filename2, dimension):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    n, m = len(df1[df1.columns[0]]), len(df2[df2.columns[0]])

    scaler = MinMaxScaler()

    df1 = scaler.fit_transform(df1)
    df2 = scaler.fit_transform(df2)

    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            sum = 0
            for k in range(0, dimension):
                sum += (df1[i-1][k] - df2[j-1][k]) ** 2

            cost = np.sqrt(sum)
            last_min = np.min(
                [dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min

    file1 = filename1.split("/")
    file1 = file1[len(file1)-1]

    file2 = filename2.split("/")
    file2 = file2[len(file2)-1]

    return file1 + "\t" + file2 + "\t" + str(dtw_matrix[n, m]/(n+m))


def fast_dtw(filename1, filename2):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    n, m = len(df1[df1.columns[0]]), len(df2[df2.columns[0]])

    scaler = MinMaxScaler()

    df1 = scaler.fit_transform(df1)
    df2 = scaler.fit_transform(df2)

    distance, path = fastdtw(df1, df2, dist=euclidean)

    return distance/(n+m)


def plot_signature(filename):
    df = pd.read_csv(filename)

    # PLOTTING
    plt.plot(df[df.columns[0]], df[df.columns[1]])
    plt.show()


def main():
    start_seconds = time.time()

    PATH = "/home/mozesbotond/WorkSpace/Signature Verification/data/MCYT/"
    FOLDER = "0000/"

    signature_list = os.listdir(PATH+FOLDER)
    signature_list.sort()

    print("Number of processors:", mp.cpu_count())

    # file1 = signature_list[0]
    # file2 = signature_list[1]

    # print("dtw: ", dtw(PATH+FOLDER+file1, PATH+FOLDER+file2, 5))
    # print("fast dtw: ", fast_dtw(PATH+FOLDER+file1, PATH+FOLDER+file2))

    pool = mp.Pool(mp.cpu_count())

    for file1 in signature_list:
        for file2 in signature_list:
            if file1 != file2:
                # result = [pool.apply(dtw, args=(
                #     PATH+FOLDER+file1, PATH+FOLDER+file2, 5))]
                print(file1 + " \t " + file2 + " \t " + str(pool.apply(fast_dtw, args=(
                    PATH+FOLDER+file1, PATH+FOLDER+file2))))

    pool.close()

    end_seconds = time.time()
    print("Elapsed time: ", end_seconds-start_seconds)


main()
