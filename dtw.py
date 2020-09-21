import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dtw(df1, df2, dimension):
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


def main():
    PATH = "~/WorkSpace/Signature Verification/data/MCYT/"

    file1 = "0000/0000f00c.csv"
    file2 = "0000/0000f05c.csv"

    df1 = pd.read_csv(PATH+file1)
    df2 = pd.read_csv(PATH+file2)

    # PLOTTING
    plt.figure(1)
    plt.subplot(211)
    plt.plot(df1[df1.columns[0]], df1[df1.columns[1]])
    plt.subplot(212)
    plt.plot(df2[df2.columns[0]], df2[df2.columns[1]])
    plt.show()

    print(dtw(df1, df2, 2))


main()
