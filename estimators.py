from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
from net import Net


class Estimatior:
    def __init__(self, net: Net):
        self.net = net

    def set_net(self, net: Net):
        self.net = net

    def get_coef(self, q_obs, p_obs):
        return None


class Est_mu(Estimatior):
    def get_coef(self, q_obs, p_obs, method="SLSQP"):
        """
        Maximum likelihood estimation for the scalar coefficient mu

        Solves minimization problem:
              sum[((q-q_obs)/sigma_q)**2] + sum[((p-p_obs)/sigma_p)**2]/ --> min
        s.t. :      A @ x == q
                    A_full.T @ (p**2) == mu * (Lambd * abs(x) * x)

        method: "SLSQP", "trust-constr"
        """
        N, M = self.net.N, self.net.M

        def fun_obj(params):
            q, p = params[:N], params[N : 2 * N]
            result = sum(((q - q_obs) / self.net.sigma_q) ** 2) + sum(
                ((p - p_obs) / self.net.sigma_p) ** 2
            )
            return result

        def constr(params):
            q = params[:N]
            p = params[N : 2 * N]
            x = params[2 * N : -1]
            mu = params[-1]

            c1 = self.net.A @ x - q[:-1]
            c2 = self.net.A_full.T @ (p**2) - mu * self.net.Lambd * abs(x) * x
            c3 = sum(q)
            return [*c1, *c2, c3]

        lb = ub = np.zeros(N + M)
        cnstrs = NonlinearConstraint(constr, lb, ub)

        _, _, x0 = self.net.calc_pressure(q_obs[:-1], p_obs[-1])
        mu0 = 1.0
        params0 = [*q_obs, *p_obs, *x0, mu0]

        params_est = minimize(fun_obj, params0, constraints=cnstrs, method=method).x

        self.q_est = params_est[:N]
        self.p_est = params_est[N : 2 * N]
        self.x_est = params_est[2 * N : -1]
        self.coef_est = params_est[-1]
        return self.coef_est


class Est_gamma(Estimatior):
    def get_coef(self, q_obs, p_obs, method="explicit"):
        """
        Maximum likelihood estimation for the vectors coefficient gamma.

        Solves minimization problem:
              sum[((q-q_obs)/sigma_q)**2] + sum[((p-p_obs)/sigma_p)**2]/ --> min
        s.t. :      | A @ x == q
                    | A_full.T @ (p**2) == gamma * (Lambd * abs(x) * x)

        Methods: "explicit", "SLSQP", "trust-constr"
        """

        if method == "explicit":
            return self.__calc_gamma_v2(q_obs, p_obs)
        else:
            return self.__calc_gamma_v1(q_obs, p_obs, method=method)

    def __calc_gamma_v1(self, q_obs, p_obs, method="SLSQP"):
        """Solution with scipy optimization"""

        N, M = self.net.N, self.net.M

        def fun_obj(params):
            q = params[:N]
            p = params[N : 2 * N]
            result = sum(((q - q_obs) / self.net.sigma_q) ** 2) + sum(
                ((p - p_obs) / self.net.sigma_p) ** 2
            )
            return result

        def constr(params):
            q = params[:N]
            p = params[N : 2 * N]
            x = params[2 * N : 2 * N + M]
            gamma = params[2 * N + M :]

            c1 = self.net.A @ x - q[:-1]
            c2 = self.net.A_full.T @ (p**2) - gamma * self.net.Lambd * abs(x) * x
            c3 = sum(q)

            return [*c1, *c2, c3]

        lb = ub = np.zeros(N + M)
        cnstrs = NonlinearConstraint(constr, lb, ub)

        _, _, x0 = self.net.calc_pressure(q_obs[:-1], p_obs[-1])
        gamma0 = np.ones(M)
        params0 = [*q_obs, *p_obs, *x0, *gamma0]

        options = {"maxiter": 10_000, "ftol": 1e-12}
        params_est = minimize(
            fun_obj,
            params0,
            constraints=cnstrs,
            method=method,
            tol=1e-12,
            options=options,
        ).x

        self.q_est = params_est[:N]
        self.p_est = params_est[N : 2 * N]
        self.x_est = params_est[2 * N : -1]
        self.mu_est = params_est[-1]
        return self.mu_est

    def __calc_gamma_v2(self, q_obs, p_obs):
        """Explicit solution wtih Lagrange multipliers method"""

        N = self.net.N
        self.q_est = self.__solve_qp(
            2 * np.identity(N), -2 * q_obs, np.ones([1, N]), np.array([0])
        )
        self.p_est = p_obs

        # if net has loops there are inf. many solutions
        # then null_space(A) can be added to rhs
        self.x_est = np.linalg.lstsq(self.net.A, self.q_est[:-1], rcond=-1)[0]
        self.gamma_est = (self.net.A_full.T @ (p_obs**2)) / (
            self.net.Lambd * np.abs(self.x_est) * self.x_est
        )

        return self.gamma_est

    def __solve_qp(self, H, c, E, d):
        """
        Solves quadratic programming problem with equality constraints
        with  Lagrange multipliers method

                0.5 * x.T @ H @ x + c.T @ x  --> min
        s. t. :      E @ x == d
        """

        Z = np.zeros([d.size, d.size])
        A = np.concatenate(
            [
                np.concatenate([H, E]),
                np.concatenate([E.T, Z]),
            ],
            axis=1,
        )
        b = np.concatenate([-c, d])
        x = np.linalg.solve(A, b)

        return x[0 : c.size]


class Est_omega(Estimatior):
    def get_coef(self, q_obs, p_obs):
        """
        Maximum likelihood estimation for the vectors coefficient omega.

        Solves minimization problem:
              sum[((q-q_obs)/sigma_q)**2] + sum[((p-p_obs)/sigma_p)**2]/ --> min
        s.t. :      | A @ x == q + omega
                    | A_full.T @ (p**2) == Lambd * abs(x) * x

        Closed form solution
        """

        self.q_est = q_obs
        self.p_est = p_obs
        tmp = self.net.A_full.T @ (self.p_est**2) / self.net.Lambd
        self.x_est = np.sign(tmp) * np.sqrt(np.abs(tmp))
        self.omega_est = self.q_est - self.net.A_full @ self.x_est

        return self.omega_est
