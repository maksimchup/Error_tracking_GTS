import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


class WilcoxonTest:
    def __init__(self, n=45, m=20, alpha=0.01):
        self.n = n
        self.m = m
        self.alpha = alpha

    def detect(self, ts):
        self.ts = ts
        self.N = ts.shape[0]
        self.err_indices = []

        for i in range(self.N - self.n):
            x = ts[i : i + self.m]
            y = ts[i + self.m : i + self.n]
            res = mannwhitneyu(x, y)

            if res.pvalue < self.alpha:
                self.err_indices.append(i + self.m)

    def plot_result(self):
        plt.plot(self.ts)
        plt.plot(self.err_indices, self.ts[self.err_indices], ".r")


class CusumTest:
    def __init__(self, climit=5, mshift=1):
        self.climit = climit
        self.mshift = mshift

        self.ilower = []
        self.iupper = []

    def detect(self, ts, tmean=25, tdev=25):
        self.ts = ts
        self.N = ts.shape[0]

        self.mean = np.mean(ts[:tmean])
        self.dev = np.std(ts[:tdev])

        self.lowersum = np.zeros(self.N)
        self.uppersum = np.zeros(self.N)
        for i, x in enumerate(self.ts[1:], start=1):
            L = self.lowersum[i - 1] + x - self.mean + self.dev * self.mshift / 2
            U = self.uppersum[i - 1] + x - self.mean - self.dev * self.mshift / 2
            self.lowersum[i] = min(0, L)
            self.uppersum[i] = max(0, U)

        self.ilower = np.where(self.lowersum > self.climit)
        self.iupper = np.where(self.uppersum > self.climit)

        return self.ilower, self.iupper, self.lowersum, self.uppersum

    def plot_result(self):

        plt.plot(self.lowersum, "r")
        plt.plot(self.uppersum, "b")

        plt.plot([-self.climit * self.dev] * self.N, "--r", linewidth=0.5)
        plt.plot([self.climit * self.dev] * self.N, "--b", linewidth=0.5)

        plt.xlim([0, self.N])
