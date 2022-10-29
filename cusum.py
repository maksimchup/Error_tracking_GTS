import numpy as np
import matplotlib.pyplot as plt


class CusumDetector:
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
