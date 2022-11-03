import numpy as np
import warnings

# todo:
# -- rename Observations class
# -- add estimation option with scipy
# -- ?maybe? include Estimator and Sample in Net


class Net:
    """
    Isothermal stationary model of the pipeline network that uses the Kirchhoff system
    q - consumption in nodes    [mln m^3/day]
    p - pressure in nodes       [MPa]
    x - flow rates              [mln m^3/day]
    D - pipeline diameters      [cm]
    L - pipeline lengths        [m]
    A, A_full, B - network graph matrices
    """

    def __init__(self, A_full: np.ndarray, B: np.ndarray):
        self.A_full = A_full
        self.B = B
        self.A = A_full[:-1, :]
        self.N, self.M = A_full.shape

    def set_params(
        self,
        L: np.ndarray,
        Lambd: np.ndarray,
        D: np.ndarray,
        sigma_p: float,
        sigma_q: float,
    ):
        self.L = L
        self.Lambd = Lambd
        self.D = D
        self.sigma_p = sigma_p
        self.sigma_q = sigma_q

    def calc_pressure(self, q: np.ndarray, p_end: float, tol=1e-9):
        """
        Calculates vectors q,p,x for given BC's

        Parameters
        ----------
        q : consumption vector at all nodes except the last one
        p : scalar value of pressure at the last node
        tol : tolerance for termination

        Returns
        -------
        q : consumption vector at all nodes (w/ last )
        p : pressure vector at all notes
        x : flow rates vector for all edges

        Notes
        -----
        Solves Kirchhoff's system of non-linear equations:
        | A @ x == q
        | A_full.T @ (p**2) == Lambd*abs(x)*x
        | B @ Lambd*abs(x)*x == 0
        Uses contour flow rates method
        """

        x = np.linalg.lstsq(self.A, q, rcond=None)[0]

        if self.B.size != 0:
            step_size = 1.0
            while step_size > tol:
                dx_ch = np.linalg.solve(
                    2 * self.B @ np.diag(self.Lambd * np.abs(x)) @ self.B.T,
                    -self.B @ (self.Lambd * np.abs(x) * x),
                )
                dx = self.B.T @ dx_ch
                x += dx
                step_size = np.linalg.norm(dx)

        pot_end = p_end * p_end
        P = np.linalg.lstsq(
            self.A.T,
            self.Lambd * np.abs(x) * x - pot_end * self.A_full[-1, :].T,
            rcond=None,
        )[0]
        p = np.sqrt(np.append(P, pot_end))
        q = np.append(q, -q.sum())
        if any(P < 0):
            raise ValueError

        return q, p, x

    def check_constraints(
        self,
        q: np.ndarray,
        p: np.ndarray,
        x: np.ndarray,
        mu=1,
        gamma=1,
        omega=0,
        tol=1e-9,
    ):
        """
        Check whether the specified parameters q, p, x are valid
        for this network according to the Kirchhoff's system
        """

        cnst1 = np.allclose(self.A_full @ x, q + omega, atol=tol)
        cnst2 = np.allclose(
            self.A_full.T @ (p**2), mu * gamma * self.Lambd * np.abs(x) * x, atol=tol
        )
        return cnst1 and cnst2


class Observations:
    def __init__(self, net: Net):
        self.net = net
        self.set_usage()
        self.set_err_fun()

    def set_usage(self, q_usage=None, p_uasge=None):
        self.q_usage = q_usage
        self.p_usage = p_uasge

        if self.q_usage is None:
            self.q_usage = [
                0.9806,
                0.978,
                0.9801,
                0.982,
                0.9835,
                0.9850,
                1.0047,
                1.0232,
                1.0350,
                1.0375,
                1.0199,
                1.0035,
                1.000,
                0.9952,
                0.9957,
                0.9860,
                1.0045,
                1.0196,
                1.0236,
                1.0278,
                1.0069,
                0.9862,
                0.9700,
                0.9693,
            ]
        if p_uasge is None:
            self.p_usage = [
                1.1334,
                1.1334,
                1.1334,
                1.1331,
                1.0874,
                1.0417,
                1.0109,
                0.9825,
                0.9222,
                0.8618,
                0.8920,
                0.9222,
                0.9517,
                0.9813,
                0.9519,
                0.9225,
                0.8915,
                0.8606,
                0.8679,
                0.8751,
                1.0034,
                1.1316,
                1.1467,
                1.1618,
            ]

    def set_err_fun(self, fun=None):
        """The error function simulates a systematic error, output should be normalized.
        Magnitude can be adjusted directly in the sample() method"""
        if fun is None:
            start = 72
            stop = 100
            self.err_fun = lambda t: (np.minimum(t, stop) - np.minimum(t, start)) / (
                stop - start
            )
        else:
            self.err_fun = fun

    def calc_real_values(self, q_bc: np.ndarray, p_bc: float, n_days=10):
        """Calculates vectors q,p for all 1-hour time steps with given BC's"""

        self.q_real = np.zeros([n_days * 24, self.net.N])
        self.p_real = np.zeros([n_days * 24, self.net.N])

        for t in range(n_days * 24):
            q = self.q_usage[t % 24] * q_bc
            p_end = self.p_usage[t % 24] * p_bc
            q, p, _ = self.net.calc_pressure(q, p_end)
            self.q_real[t, :] = q
            self.p_real[t, :] = p

    def sample(self, err_type="p", err_node=10, err_magnitude=0.1):
        """Samples observations from real values and noise"""

        size = self.q_real.shape

        # random error
        noise_q = self.net.sigma_q * np.random.randn(*size)
        noise_p = self.net.sigma_p * np.random.randn(*size)

        q_obs = self.q_real + noise_q
        p_obs = self.p_real + noise_p

        # systematic error
        sys_error = err_magnitude * self.err_fun(np.arange(size[0]))
        if err_type.lower() == "p":
            p_obs[:, err_node] = p_obs[:, err_node] + sys_error
        elif err_type.lower() == "q":
            q_obs[:, err_node] = q_obs[:, err_node] + sys_error
        else:
            warnings.warn(
                "Error type should be either 'p' - pressure, 'q' -  consumption."
            )

        return q_obs, p_obs

    def set_observations(self, q_obs, p_obs):
        """Use it only for real measurments, for simulation use sample()"""
        self.q_obs = q_obs
        self.p_obs = p_obs
