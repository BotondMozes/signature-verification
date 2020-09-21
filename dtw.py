import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dtw(filename1, filename2, dimension):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    n, m = len(df1[df1.columns[0]]), len(df2[df2.columns[0]])

    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            sum = 0
            for k in range(0, dimension):
                index = df1.columns[k]
                sum += (df1[index][i-1] - df2[index][j-1]) ** 2

            cost = np.sqrt(sum)
            last_min = np.min(
                [dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m]/(n+m)


def plot_signature(filename):
    df = pd.read_csv(filename)

    # PLOTTING
    plt.plot(df[df.columns[0]], df[df.columns[1]])
    plt.show()


def main():
    PATH = "./WorkSpace/Signature Verification/data/MCYT/"
    FOLDER = "0000/"

    signature_list = os.listdir(PATH+FOLDER)

    for file1 in signature_list:
        for file2 in signature_list:
            print(file1 + "\t" + file2 + "\t" +
                  str(dtw(PATH+FOLDER+file1, PATH+FOLDER+file2, 5)))


main()
