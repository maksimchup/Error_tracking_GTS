from net import Net, Observations, Estimatior
import numpy as np
import matplotlib.pyplot as plt
import pickle

net1 = pickle.load(open("net1.p", "br"))

q_bc = np.array(
    [
        -0.7367,
        -4.2121,
        -0.0424,
        -1.9237,
        -1.9254,
        -2.6563,
        0.0,
        0.0,
        0.0,
        6.8244,
    ]
)
p_bc = 3.3139
n_days = 10
obs = Observations(net1)
obs.calc_real_values(q_bc, p_bc, n_days)
q_obs, p_obs = obs.sample(err_type="p", err_node=10, err_magnitude=0.1)

est = Estimatior(net1)
gamma_est = np.zeros([24 * n_days, 12])

for t in range(24 * n_days):
    gamma_est[t, :] = est.calc_gamma(q_obs[t], p_obs[t])

for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.title(f"Pipeline {i+1}")
    plt.plot(gamma_est[:, i])

plt.show()
