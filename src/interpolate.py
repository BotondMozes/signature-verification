import settings
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate


def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def interpolate_mcyt_file(inputfile, outputfile):
    # print(outputfile)
    df = pd.read_csv(inputfile, usecols=settings.MCYT_FIELDS)

    x = df['X']
    y = df[' Y']
    p = df[' P']
    N = len(df.index)

    # print("N: "+str(N))
    t = np.arange(0, N)
    dt = t[N-1]/settings.LENGTH
    tnew = np.arange(0, t[N-1], dt)
    # print(len(tnew))
    fx = interpolate.interp1d(t, x)
    fy = interpolate.interp1d(t, y)
    fp = interpolate.interp1d(t, p)

    xnew = fx(tnew)   # use interpolation function returned by `interp1d`
    ynew = fy(tnew)   # use interpolation function returned by `interp1d`
    pnew = fp(tnew)

    d = {'t': tnew, 'x': xnew, 'y': ynew, 'p': pnew}
    df = pd.DataFrame(data=d)
    df.to_csv(outputfile, index=False)
    # plt.plot(xnew, 1000-ynew)
    # plt.show()
    return


def interpolate_mobisig_file(inputfile, outputfile):
    # print(inputfile)
    df = pd.read_csv(inputfile, usecols=[0, 1, 2, 3])

    x = df['x']
    y = df['y']
    t = df['timestamp']
    p = df['pressure']

    t = t - t[0]
    N = len(df.index)
    # print("N: "+str(N))
    t = np.arange(0, N)
    dt = t[N-1]/settings.LENGTH
    tnew = np.arange(0, t[N-1], dt)
    # print(len(tnew))
    fx = interpolate.interp1d(t, x)
    fy = interpolate.interp1d(t, y)
    fp = interpolate.interp1d(t, p)

    xnew = fx(tnew)   # use interpolation function returned by `interp1d`
    ynew = fy(tnew)   # use interpolation function returned by `interp1d`
    pnew = fp(tnew)

    d = {'t': tnew, 'x': xnew, 'y': ynew, 'p': pnew}
    df = pd.DataFrame(data=d)
    df.to_csv(outputfile, index=False)
    # plt.plot(xnew, 1000-ynew)
    # plt.show()
    return


def interpolate_all():
    # MCYT
    folder_list = listdirs(settings.MCYT_DATA)

    for folder in folder_list:
        signature_list = os.listdir(settings.MCYT_DATA + folder)
        signature_list.sort()

        if len(signature_list) != 0:
            for file1 in signature_list:
                interpolate_mcyt_file(
                    settings.MCYT_DATA+folder+"/"+file1, settings.MCYT_INTERP+folder+"/"+file1)

    # MOBISIG
    folder_list = listdirs(settings.MOBISIG_DATA)

    for folder in folder_list:
        signature_list = os.listdir(settings.MOBISIG_DATA + folder)
        signature_list.sort()

        if len(signature_list) != 0:
            for file1 in signature_list:
                interpolate_mobisig_file(
                    settings.MOBISIG_DATA+folder+"/"+file1, settings.MOBISIG_INTERP+folder+"/"+file1)


if __name__ == '__main__':
    interpolate_all()
