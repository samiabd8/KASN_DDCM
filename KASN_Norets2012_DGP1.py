#!/usr/bin/env python3

from __future__ import annotations

import sys
import argparse
import time
import copy
import math
from typing import Callable

import numpy as np
import scipy.stats

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# 1.  Exact DDCM solution (Norets, 2012)
# ---------------------------------------------------------------------------

EULER_MASCHERONI = 0.5772156649015329

class BusEngineEVIID:
    """Exact dynamic program solver for the bus engine model with EV‑IID shocks."""
    INPUT_DIM = 7  # (x, alpha1, alpha2, rho, eta1, eta2, eta3)

    def __init__(self, M: int = 90):
        if M < 5:
            raise ValueError("M must be at least 5.")
        self.M = int(M)

    def solve_ev(self, alpha1: float, alpha2: float, rho: float,
                 eta: np.ndarray, tol: float = 1e-10,
                 max_iter: int = 10_000,
                 ev_init: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        eta = np.asarray(eta, dtype=np.float64)
        if eta.shape != (3,):
            raise ValueError(f"eta must have shape (3,), got {eta.shape}.")
        if not np.isclose(eta.sum(), 1.0, atol=1e-6):
            raise ValueError(f"eta must sum to 1, got {eta.sum():.6f}")
        if not (0.0 <= rho < 1.0):
            raise ValueError(f"rho must be in [0, 1), got {rho}")

        if ev_init is None:
            EV = np.zeros(self.M, dtype=np.float64)
        else:
            EV = np.asarray(ev_init, dtype=np.float64).copy()
            if EV.shape != (self.M,):
                raise ValueError("ev_init has wrong shape.")
        EV_new = np.empty_like(EV)

        for it in range(max_iter):
            self._bellman_step(EV, EV_new, alpha1, alpha2, rho, eta)
            err = np.max(np.abs(EV_new - EV))
            EV[:] = EV_new
            if err < tol:
                return EV, it + 1
        return EV, max_iter

    def _bellman_step(self, EV: np.ndarray, EV_new: np.ndarray,
                      alpha1: float, alpha2: float, rho: float,
                      eta: np.ndarray) -> None:
        M = self.M
        EV_p1 = np.empty_like(EV)
        EV_p1[:-1] = EV[1:]
        EV_p1[-1] = EV[-1]
        EV_p2 = np.empty_like(EV)
        EV_p2[:-2] = EV[2:]
        EV_p2[-2:] = EV[-1]
        ev_cont_1 = eta[0] * EV + eta[1] * EV_p1 + eta[2] * EV_p2
        ev_cont_2 = eta[0] * EV[0] + eta[1] * EV[1] + eta[2] * EV[2]
        x = np.arange(1, M + 1, dtype=np.float64)
        v1 = alpha1 * x + rho * ev_cont_1
        v2 = alpha2 + rho * ev_cont_2
        v1[-1] = -np.inf
        v_max = np.maximum(v1, v2)
        EV_new[:] = (v_max +
                     np.log(np.exp(v1 - v_max) + np.exp(v2 - v_max)) +
                     EULER_MASCHERONI)

    def compute_F(self, EV: np.ndarray, rho: float, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        EV_p1 = np.empty_like(EV)
        EV_p1[:-1] = EV[1:]
        EV_p1[-1] = EV[-1]
        EV_p2 = np.empty_like(EV)
        EV_p2[:-2] = EV[2:]
        EV_p2[-2:] = EV[-1]
        ev_cont = eta[0] * EV + eta[1] * EV_p1 + eta[2] * EV_p2
        EV_2 = rho * (eta[0] * EV[0] + eta[1] * EV[1] + eta[2] * EV[2])
        return rho * ev_cont - EV_2

    def simulate_panel(self, alpha1: float, alpha2: float, rho: float,
                       eta: np.ndarray, n_buses: int, n_periods: int,
                       rng: np.random.Generator,
                       x_init: np.ndarray | None = None) -> dict:
        eta = np.asarray(eta, dtype=np.float64)
        EV, _ = self.solve_ev(alpha1, alpha2, rho, eta)
        M = self.M

        EV_p1 = np.empty_like(EV)
        EV_p1[:-1] = EV[1:]
        EV_p1[-1] = EV[-1]
        EV_p2 = np.empty_like(EV)
        EV_p2[:-2] = EV[2:]
        EV_p2[-2:] = EV[-1]
        ev_cont_1 = eta[0] * EV + eta[1] * EV_p1 + eta[2] * EV_p2
        ev_cont_2 = eta[0] * EV[0] + eta[1] * EV[1] + eta[2] * EV[2]
        x_grid = np.arange(1, M + 1, dtype=np.float64)
        v1 = alpha1 * x_grid + rho * ev_cont_1
        v2 = alpha2 + rho * ev_cont_2
        v1[-1] = -np.inf
        v_max = np.maximum(v1, v2)
        e1 = np.exp(v1 - v_max)
        e2 = np.exp(v2 - v_max)
        P2 = e2 / (e1 + e2)

        x = np.empty((n_periods, n_buses), dtype=np.int64)
        d = np.empty((n_periods, n_buses), dtype=np.int64)
        if x_init is None:
            x[0] = 1
        else:
            x[0] = np.asarray(x_init, dtype=np.int64)

        for t in range(n_periods):
            u = rng.uniform(size=n_buses)
            d[t] = np.where(u < P2[x[t] - 1], 2, 1)
            if t < n_periods - 1:
                jumps = rng.choice(3, p=eta, size=n_buses)
                post = np.where(d[t] == 1, x[t], 1)
                x[t + 1] = np.minimum(post + jumps, M)

        return {"x": x, "d": d}

    def compute_ccp_replace(self, EV: np.ndarray, alpha1: float,
                            alpha2: float, rho: float,
                            eta: np.ndarray) -> np.ndarray:
        M = self.M
        EV_p1 = np.empty_like(EV)
        EV_p1[:-1] = EV[1:]
        EV_p1[-1] = EV[-1]
        EV_p2 = np.empty_like(EV)
        EV_p2[:-2] = EV[2:]
        EV_p2[-2:] = EV[-1]
        ev_cont_1 = eta[0] * EV + eta[1] * EV_p1 + eta[2] * EV_p2
        ev_cont_2 = eta[0] * EV[0] + eta[1] * EV[1] + eta[2] * EV[2]
        x = np.arange(1, M + 1, dtype=np.float64)
        v1 = alpha1 * x + rho * ev_cont_1
        v2 = alpha2 + rho * ev_cont_2
        v1[-1] = -np.inf
        v_max = np.maximum(v1, v2)
        e1 = np.exp(v1 - v_max)
        e2 = np.exp(v2 - v_max)
        return e2 / (e1 + e2)


# ---------------------------------------------------------------------------
# 2.  MCMC: prior, likelihood, sampler
# ---------------------------------------------------------------------------

class MCMCPrior:
    def __init__(self,
                 alpha1_lo: float = -0.006, alpha1_hi: float = 0.0,
                 alpha2_lo: float = -25.0, alpha2_hi: float = -5.0,
                 rho_lo: float = 0.0, rho_hi: float = 0.99,
                 eta_concentration: tuple[float, float, float] = (34.0, 64.0, 2.0)):
        self.alpha1_lo, self.alpha1_hi = alpha1_lo, alpha1_hi
        self.alpha2_lo, self.alpha2_hi = alpha2_lo, alpha2_hi
        self.rho_lo, self.rho_hi = rho_lo, rho_hi
        self.eta_alpha = np.asarray(eta_concentration, dtype=np.float64)

    def in_support(self, alpha1: float, alpha2: float, rho: float,
                   eta: np.ndarray) -> bool:
        return bool(
            self.alpha1_lo <= alpha1 <= self.alpha1_hi and
            self.alpha2_lo <= alpha2 <= self.alpha2_hi and
            self.rho_lo <= rho <= self.rho_hi and
            np.all(eta > 0.0) and abs(eta.sum() - 1.0) < 1e-9
        )

    def log_pdf(self, alpha1: float, alpha2: float, rho: float,
                eta: np.ndarray) -> float:
        if not self.in_support(alpha1, alpha2, rho, eta):
            return -np.inf
        return float(scipy.stats.dirichlet.logpdf(eta, self.eta_alpha))

    def sample(self, rng: np.random.Generator):
        return (
            float(rng.uniform(self.alpha1_lo, self.alpha1_hi)),
            float(rng.uniform(self.alpha2_lo, self.alpha2_hi)),
            float(rng.uniform(self.rho_lo, self.rho_hi)),
            rng.dirichlet(self.eta_alpha),
        )

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))

class BusEngineLikelihood:
    def __init__(self, panel_x: np.ndarray, panel_d: np.ndarray, M: int = 90):
        self.x = np.asarray(panel_x, dtype=np.int64)
        self.d = np.asarray(panel_d, dtype=np.int64)
        self.M = int(M)
        self.T, self.N = self.x.shape

        if not np.all((self.x >= 1) & (self.x <= M)):
            raise ValueError("panel_x entries must lie in {1, …, M}.")
        if not np.all((self.d == 1) | (self.d == 2)):
            raise ValueError("panel_d entries must be 1 or 2.")
        if ((self.x == M) & (self.d == 1)).any():
            raise ValueError("Inconsistent: d=1 observed at boundary x=M.")

        post_state = np.where(self.d == 1, self.x, 1)
        self._post_prev = post_state[:-1]
        self._x_next = self.x[1:]

        self._x_flat = self.x.ravel()
        self._d_flat = self.d.ravel()
        self._mask_M = (self._x_flat == M)

    def log_likelihood(self, alpha1: float, alpha2: float, rho: float,
                       eta: np.ndarray, F: np.ndarray) -> float:
        z = alpha1 * self._x_flat - alpha2 + F[self._x_flat - 1]
        log_p1 = -_softplus(-z)
        log_p2 = -_softplus(z)
        log_p_choice = np.where(self._d_flat == 1, log_p1, log_p2)
        log_p_choice[self._mask_M] = 0.0
        ll_choice = float(log_p_choice.sum())

        prob_trans = np.zeros_like(self._x_next, dtype=np.float64)
        for j in range(3):
            target = np.minimum(self._post_prev + j, self.M)
            prob_trans += eta[j] * (target == self._x_next)
        prob_trans = np.maximum(prob_trans, 1e-300)
        ll_trans = float(np.log(prob_trans).sum())
        return ll_choice + ll_trans


FOracle = Callable[[float, float, float, np.ndarray], np.ndarray]

def make_F_oracle_exact(bus: BusEngineEVIID) -> FOracle:
    def F_fn(alpha1, alpha2, rho, eta):
        EV, _ = bus.solve_ev(alpha1, alpha2, rho, eta)
        return bus.compute_F(EV, rho, eta)
    return F_fn

def make_F_oracle_kasn(model, M: int = 90) -> FOracle:
    x_grid = np.arange(1, M + 1, dtype=np.float64)
    template = np.empty((M, 7), dtype=np.float64)
    template[:, 0] = x_grid

    def F_fn(alpha1, alpha2, rho, eta):
        X = template.copy()
        X[:, 1] = alpha1
        X[:, 2] = alpha2
        X[:, 3] = rho
        X[:, 4:7] = eta
        return model.predict(X)
    return F_fn


def alr(eta: np.ndarray) -> np.ndarray:
    return np.log(eta[:-1] / eta[-1])

def alr_inv(y: np.ndarray) -> np.ndarray:
    e = np.exp(y)
    s = 1.0 + e.sum()
    eta = np.empty(3, dtype=np.float64)
    eta[:2] = e / s
    eta[2] = 1.0 / s
    return eta

def log_jacobian_alr(eta: np.ndarray) -> float:
    return float(np.log(eta).sum())


class BusEngineMCMC:
    def __init__(self, likelihood: BusEngineLikelihood, prior: MCMCPrior,
                 F_oracle: FOracle,
                 sigma_alpha: tuple[float, float] = (8e-4, 2.0),
                 sigma_rho: float = 0.04,
                 sigma_eta: float = 0.15,
                 target_acc: float = 0.40,
                 adapt_window: int = 50):
        self.lik = likelihood
        self.prior = prior
        self.F_oracle = F_oracle
        self.sigma_alpha = np.asarray(sigma_alpha, dtype=np.float64)
        self.sigma_rho = float(sigma_rho)
        self.sigma_eta = float(sigma_eta)
        self.target_acc = float(target_acc)
        self.adapt_window = int(adapt_window)

    def _ll_lp(self, alpha1, alpha2, rho, eta):
        log_p = self.prior.log_pdf(alpha1, alpha2, rho, eta)
        if not np.isfinite(log_p):
            return -np.inf, log_p
        F = self.F_oracle(alpha1, alpha2, rho, eta)
        if not np.all(np.isfinite(F)):
            return -np.inf, log_p
        log_l = self.lik.log_likelihood(alpha1, alpha2, rho, eta, F)
        return log_l, log_p

    def sample(self, n_iter: int, n_burn: int,
               init_theta=None, rng: np.random.Generator | None = None,
               verbose: bool = True) -> tuple[dict, dict]:
        rng = rng or np.random.default_rng()
        if init_theta is None:
            for _ in range(50):
                alpha1, alpha2, rho, eta = self.prior.sample(rng)
                log_l, log_p = self._ll_lp(alpha1, alpha2, rho, eta)
                if np.isfinite(log_l) and np.isfinite(log_p):
                    break
            else:
                raise RuntimeError("Failed to find an interior init theta.")
        else:
            alpha1, alpha2, rho, eta = init_theta
            log_l, log_p = self._ll_lp(alpha1, alpha2, rho, eta)
            if not (np.isfinite(log_l) and np.isfinite(log_p)):
                raise ValueError("init_theta yields -inf log‑posterior.")
        log_j = log_jacobian_alr(eta)

        n_total = n_burn + n_iter
        chain_a1 = np.empty(n_total)
        chain_a2 = np.empty(n_total)
        chain_r = np.empty(n_total)
        chain_e = np.empty((n_total, 3))
        chain_lp = np.empty(n_total)

        sigma_a = self.sigma_alpha.copy()
        sigma_r = self.sigma_rho
        sigma_e = self.sigma_eta

        acc_a_total = acc_r_total = acc_e_total = 0
        win_a = win_r = win_e = 0
        win_n = 0

        for it in range(n_total):
            a1_p = alpha1 + sigma_a[0] * rng.standard_normal()
            a2_p = alpha2 + sigma_a[1] * rng.standard_normal()
            log_l_p, log_p_p = self._ll_lp(a1_p, a2_p, rho, eta)
            log_alpha = (log_l_p + log_p_p) - (log_l + log_p)
            if np.log(rng.uniform()) < log_alpha:
                alpha1, alpha2 = a1_p, a2_p
                log_l, log_p = log_l_p, log_p_p
                acc_a_total += 1
                win_a += 1

            r_p = rho + sigma_r * rng.standard_normal()
            log_l_p, log_p_p = self._ll_lp(alpha1, alpha2, r_p, eta)
            log_alpha = (log_l_p + log_p_p) - (log_l + log_p)
            if np.log(rng.uniform()) < log_alpha:
                rho = r_p
                log_l, log_p = log_l_p, log_p_p
                acc_r_total += 1
                win_r += 1

            y_curr = alr(eta)
            y_p = y_curr + sigma_e * rng.standard_normal(2)
            eta_p = alr_inv(y_p)
            log_l_p, log_p_p = self._ll_lp(alpha1, alpha2, rho, eta_p)
            log_j_p = log_jacobian_alr(eta_p)
            log_alpha = ((log_l_p + log_p_p + log_j_p) -
                         (log_l + log_p + log_j))
            if np.log(rng.uniform()) < log_alpha:
                eta = eta_p
                log_l, log_p, log_j = log_l_p, log_p_p, log_j_p
                acc_e_total += 1
                win_e += 1

            chain_a1[it] = alpha1
            chain_a2[it] = alpha2
            chain_r[it] = rho
            chain_e[it] = eta
            chain_lp[it] = log_l + log_p
            win_n += 1

            if it < n_burn and win_n >= self.adapt_window:
                a_a = win_a / win_n
                a_r = win_r / win_n
                a_e = win_e / win_n
                sigma_a *= np.exp(0.5 * (a_a - self.target_acc))
                sigma_r *= np.exp(0.5 * (a_r - self.target_acc))
                sigma_e *= np.exp(0.5 * (a_e - self.target_acc))
                if verbose:
                    print(f"  [it {it+1:5d} burn] "
                          f"acc(α,ρ,η)=({a_a:.2f}, {a_r:.2f}, {a_e:.2f})  "
                          f"σ=({sigma_a[0]:.4f}, {sigma_a[1]:.3f}, "
                          f"{sigma_r:.3f}, {sigma_e:.3f})")
                win_a = win_r = win_e = 0
                win_n = 0

            if verbose and (it + 1) % max(1, n_total // 10) == 0:
                phase = "burn" if it < n_burn else "samp"
                print(f"  [{phase} {it+1:5d}/{n_total}] log_post={log_l+log_p:.2f}  "
                      f"theta=({alpha1:+.4f}, {alpha2:+.2f}, {rho:.3f}, "
                      f"η=[{eta[0]:.3f}, {eta[1]:.3f}, {eta[2]:.3f}])")

        chain = {
            "alpha1": chain_a1[n_burn:],
            "alpha2": chain_a2[n_burn:],
            "rho": chain_r[n_burn:],
            "eta": chain_e[n_burn:],
            "log_post": chain_lp[n_burn:],
        }
        diag = {
            "acc_alpha": acc_a_total / n_total,
            "acc_rho": acc_r_total / n_total,
            "acc_eta": acc_e_total / n_total,
            "sigma_alpha": sigma_a,
            "sigma_rho": sigma_r,
            "sigma_eta": sigma_e,
        }
        return chain, diag


# ---------------------------------------------------------------------------
# 3.  KASN implementation 
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class EmpiricalCDFTransformer:
        """Transforms features to empirical CDF in [0,1]."""
        def __init__(self):
            self.sorted_values_ = None
            self.n_train_ = None
        def fit(self, X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.sorted_values_ = np.sort(X, axis=0)
            self.n_train_ = X.shape[0]
            return self
        def transform(self, X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n = self.n_train_
            cdf = np.zeros_like(X, dtype=np.float64)
            for i in range(X.shape[1]):
                cdf[:, i] = np.searchsorted(
                    self.sorted_values_[:, i], X[:, i], side='right') / (n + 1.0)
            return np.clip(cdf, 0.0, 1.0)
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

    class BSplineBasis(nn.Module):
        def __init__(self, in_features, grid_size=5, spline_order=4, grid_range=[-0.5, 1.5]):
            super().__init__()
            self.in_features = in_features
            self.grid_size = grid_size
            self.spline_order = spline_order
            self.register_buffer("grid", self._create_grid(grid_range, grid_size))
        def _create_grid(self, grid_range, grid_size):
            h = (grid_range[1] - grid_range[0]) / grid_size
            g = torch.arange(-self.spline_order, grid_size + self.spline_order + 1) * h + grid_range[0]
            return g.expand(self.in_features, -1).contiguous()
        def b_splines(self, x):
            grid = self.grid
            x = x.unsqueeze(-1)
            bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
            for k in range(1, self.spline_order + 1):
                left = (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]).clamp_min(1e-8)
                right = (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:(-k)]).clamp_min(1e-8)
                bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
            return bases.contiguous()
        def forward(self, x):
            s = x.shape
            return self.b_splines(x.reshape(-1, self.in_features)).reshape(*s[:-1], -1)

    class KASNLayer(nn.Module):
        def __init__(self, in_features, out_features, grid_size=5,
                     spline_order=4, base_activation=nn.SiLU,
                     grid_range=[-0.5, 1.5], use_residual=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.use_residual = use_residual and (in_features == out_features)
            self.basis = BSplineBasis(in_features, grid_size, spline_order, grid_range)
            self.num_basis = grid_size + spline_order
            self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, self.num_basis))
            self.w_b = nn.Parameter(torch.zeros(1))
            self.w_s = nn.Parameter(torch.ones(1))
            self.w_b.requires_grad = False
            self.w_s.requires_grad = False
            self.base_activation = base_activation()
            self.grid_size = grid_size
            self.reset_parameters()
        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            nn.init.uniform_(self.spline_weight, -0.5/self.grid_size, 0.5/self.grid_size)
            with torch.no_grad():
                self.w_b.fill_(0.0)
                self.w_s.fill_(1.0)
        def forward(self, x):
            base_out = F.linear(self.base_activation(x), self.base_weight)
            spline_out = F.linear(self.basis(x).view(x.size(0), -1),
                                  self.spline_weight.view(self.out_features, -1))
            out = self.w_b * base_out + self.w_s * spline_out
            return out + x if self.use_residual else out
        def group_lasso_regularization_loss(self, gl):
            return gl * torch.norm(self.spline_weight, p=2, dim=2).sum()

    class KASN(nn.Module):
        def __init__(self, input_dim, n_samples, gamma=0.4, kasn_width=None, depth=3,
                     dual_reg=False, kart_s=4, prune_during_training=False):
            super().__init__()
            self.input_dim = input_dim
            self.n_samples = n_samples
            self.gamma = gamma
            self.dual_reg = dual_reg
            self.kart_s = kart_s
            self.prune_during_training = prune_during_training

            self.G = max(5, int(n_samples ** gamma))
            self.L = depth if depth is not None else max(3, int(np.log(n_samples)))
            self.W = kasn_width if kasn_width is not None else (2 * kart_s + 1)

            self.layers = nn.ModuleList()
            self.layers.append(KASNLayer(input_dim, self.W, grid_size=self.G, use_residual=False))
            for _ in range(self.L - 2):
                self.layers.append(KASNLayer(self.W, self.W, grid_size=self.G, use_residual=True))
            self.layers.append(KASNLayer(self.W, 1, grid_size=self.G, use_residual=False))

            self.scaler_X = EmpiricalCDFTransformer()
            self.scaler_y = StandardScaler()

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def group_lasso_regularization_loss(self, gl):
            return sum(l.group_lasso_regularization_loss(gl) for l in self.layers)

        def compute_lambda_reg(self):
            return np.sqrt(np.log(self.L * self.W**2 + 1e-8) / self.n_samples)

        def fit(self, X_train_np, y_train_np, X_val_np, y_val_np,
                epochs=1000, lr=1e-3, batch_size=256, patience=200,
                weight_decay=1e-12, l1_reg_scale=0.0, group_lasso_reg_scale=1e-3):
            if y_train_np.ndim == 1:
                y_train_np = y_train_np.reshape(-1, 1)
            if y_val_np.ndim == 1:
                y_val_np = y_val_np.reshape(-1, 1)

            self.scaler_X.fit(X_train_np)
            self.scaler_y.fit(y_train_np)

            X_train_t = torch.tensor(self.scaler_X.transform(X_train_np), dtype=torch.float32)
            y_train_t = torch.tensor(self.scaler_y.transform(y_train_np), dtype=torch.float32)
            X_val_t = torch.tensor(self.scaler_X.transform(X_val_np), dtype=torch.float32)
            y_val_t = torch.tensor(self.scaler_y.transform(y_val_np), dtype=torch.float32)

            pin = (torch.cuda.is_available())
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      pin_memory=pin, num_workers=4 if pin else 0)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(device)
            X_val_gpu = X_val_t.to(device)
            y_val_gpu = y_val_t.to(device)

            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
            criterion = nn.MSELoss()
            lambda_reg = self.compute_lambda_reg()
            amp_enabled = (device.type == 'cuda')
            scaler_amp = torch.amp.GradScaler('cuda', enabled=amp_enabled)

            best_val_loss = float('inf')
            best_state = None
            best_epoch = 0
            epochs_no_improve = 0

            for epoch in range(epochs):
                self.train()
                epoch_mse = 0.0
                for Xb, yb in train_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        pred = self(Xb)
                        mse_loss = criterion(pred, yb)
                        gl_loss = self.group_lasso_regularization_loss(group_lasso_reg_scale)
                        total_loss = mse_loss + lambda_reg * gl_loss
                    scaler_amp.scale(total_loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 50.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    epoch_mse += mse_loss.item() * Xb.size(0)
                scheduler.step()
                avg_mse = epoch_mse / len(X_train_t)

                self.eval()
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        val_pred = self(X_val_gpu)
                        val_loss = criterion(val_pred, y_val_gpu).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch % 50 == 0 or epoch == epochs - 1:
                    print(f'KASN Epoch {epoch:4d}/{epochs}: TrainMSE={avg_mse:.6f}, ValMSE={val_loss:.6f}')
                if patience and epochs_no_improve >= patience:
                    print(f'  Early stopping at epoch {epoch} (best {best_epoch}, val {best_val_loss:.6f})')
                    break

            if best_state is not None:
                self.load_state_dict(best_state)
            return None, None, best_val_loss

        def predict(self, X_np):
            self.eval()
            device = next(self.parameters()).device
            X_t = torch.tensor(self.scaler_X.transform(X_np), dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = self(X_t).cpu().numpy()
            return self.scaler_y.inverse_transform(pred).flatten()


# ---------------------------------------------------------------------------
# 4.  Validation helpers (training data, plotting, summary)
# ---------------------------------------------------------------------------

class ParameterPriorBox:
    def __init__(self,
                 alpha1_lo=-0.006, alpha1_hi=0.0,
                 alpha2_lo=-25.0, alpha2_hi=-5.0,
                 rho_lo=0.0, rho_hi=0.99,
                 eta_alpha=(34.0, 64.0, 2.0)):
        self.alpha1_lo, self.alpha1_hi = alpha1_lo, alpha1_hi
        self.alpha2_lo, self.alpha2_hi = alpha2_lo, alpha2_hi
        self.rho_lo, self.rho_hi = rho_lo, rho_hi
        self.eta_alpha = np.asarray(eta_alpha)

    def sample(self, rng: np.random.Generator):
        alpha1 = rng.uniform(self.alpha1_lo, self.alpha1_hi)
        alpha2 = rng.uniform(self.alpha2_lo, self.alpha2_hi)
        rho = rng.uniform(self.rho_lo, self.rho_hi)
        eta = rng.dirichlet(self.eta_alpha)
        return alpha1, alpha2, rho, eta

def make_train_val_test(bus: BusEngineEVIID, prior: ParameterPriorBox,
                        n_train: int, n_val: int, n_test: int,
                        rng: np.random.Generator, verbose: bool = True):
    M = bus.M
    n_total = n_train + n_val + n_test
    X = np.empty((n_total, bus.INPUT_DIM), dtype=np.float64)
    y = np.empty(n_total, dtype=np.float64)
    F_all = np.empty((M,), dtype=np.float64)

    for i in range(n_total):
        alpha1, alpha2, rho, eta = prior.sample(rng)
        EV, _ = bus.solve_ev(alpha1, alpha2, rho, eta)
        F_all[:] = bus.compute_F(EV, rho, eta)
        x = rng.integers(1, M + 1)
        X[i, 0] = x
        X[i, 1] = alpha1
        X[i, 2] = alpha2
        X[i, 3] = rho
        X[i, 4:7] = eta
        y[i] = F_all[x - 1]

    return {
        "X_train": X[:n_train], "y_train": y[:n_train],
        "X_val":   X[n_train:n_train+n_val], "y_val":   y[n_train:n_train+n_val],
        "X_test":  X[n_train+n_val:],        "y_test":  y[n_train+n_val:],
    }

def _plot_posteriors(chain_e, chain_k, truth, out_path, title_extra=""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    def _hist(ax, vals_e, vals_k, true_val, label):
        ax.hist(vals_e, bins=40, density=True, alpha=0.55,
                color="steelblue", label="exact $F$")
        ax.hist(vals_k, bins=40, density=True, alpha=0.55,
                color="darkorange", label="KASN $\\hat F$")
        ax.axvline(true_val, color="red", lw=1.4, label="truth")
        ax.set_xlabel(label); ax.set_ylabel("density")
        ax.legend(fontsize=8)
    _hist(axes[0, 0], chain_e["alpha1"], chain_k["alpha1"], truth["alpha1"], "$\\alpha_1$")
    _hist(axes[0, 1], chain_e["alpha2"], chain_k["alpha2"], truth["alpha2"], "$\\alpha_2$")
    _hist(axes[0, 2], chain_e["rho"],    chain_k["rho"],    truth["rho"],    "$\\rho$")
    for k in range(3):
        _hist(axes[1, k], chain_e["eta"][:, k], chain_k["eta"][:, k],
              truth["eta"][k], f"$\\eta_{k+1}$")
    fig.suptitle(f"Posterior comparison: exact-$F$ vs KASN-$\\hat F$ {title_extra}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def _plot_traces(chain_e, chain_k, truth, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    rows = [("alpha1", truth["alpha1"], "$\\alpha_1$"),
            ("alpha2", truth["alpha2"], "$\\alpha_2$"),
            ("rho",    truth["rho"],    "$\\rho$")]
    for ax, (key, true_val, lbl) in zip(axes[0], rows):
        ax.plot(chain_e[key], lw=0.4, color="steelblue", label="exact")
        ax.plot(chain_k[key], lw=0.4, color="darkorange", alpha=0.8, label="KASN")
        ax.axhline(true_val, color="red", lw=0.7)
        ax.set_ylabel(lbl); ax.legend(fontsize=7)
    for k in range(3):
        ax = axes[1, k]
        ax.plot(chain_e["eta"][:, k], lw=0.4, color="steelblue")
        ax.plot(chain_k["eta"][:, k], lw=0.4, color="darkorange", alpha=0.8)
        ax.axhline(truth["eta"][k], color="red", lw=0.7)
        ax.set_ylabel(f"$\\eta_{k+1}$")
    axes[1, 1].set_xlabel("MCMC iteration (post burn‑in)")
    fig.suptitle("Trace plots", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def _print_summary(chain_e, chain_k, truth):
    print("\n" + "=" * 86)
    print("POSTERIOR SUMMARIES   (|Δmean(KASN, exact)| / sd_exact in last col)")
    print("=" * 86)
    print(f"{'param':<10} {'truth':>10} {'exact mean':>12} {'exact sd':>10}  "
          f"{'KASN mean':>12} {'KASN sd':>10}  {'|Δ|/σ':>8}")
    print("-" * 86)
    for key in ("alpha1", "alpha2", "rho"):
        e_m, e_s = chain_e[key].mean(), chain_e[key].std()
        k_m, k_s = chain_k[key].mean(), chain_k[key].std()
        bias = abs(k_m - e_m) / (e_s + 1e-12)
        print(f"{key:<10} {truth[key]:+10.4f} {e_m:+12.4f} {e_s:10.4f}  "
              f"{k_m:+12.4f} {k_s:10.4f}  {bias:8.3f}")
    for k in range(3):
        e_m, e_s = chain_e["eta"][:, k].mean(), chain_e["eta"][:, k].std()
        k_m, k_s = chain_k["eta"][:, k].mean(), chain_k["eta"][:, k].std()
        bias = abs(k_m - e_m) / (e_s + 1e-12)
        print(f"eta_{k+1:<7} {truth['eta'][k]:+10.4f} {e_m:+12.4f} {e_s:10.4f}  "
              f"{k_m:+12.4f} {k_s:10.4f}  {bias:8.3f}")


# ---------------------------------------------------------------------------
# 5.  __main__ and Mode Select
# ---------------------------------------------------------------------------

def _run_self_check():
    print("=" * 60)
    print("SELF‑CHECK: exact model and exact‑F MCMC (no KASN required)")
    print("=" * 60)
    bus = BusEngineEVIID(M=90)
    eta = np.array([0.34, 0.64, 0.02])
    EV, n_iter = bus.solve_ev(alpha1=-0.003, alpha2=-10.0, rho=0.95, eta=eta)
    print(f"DP converged in {n_iter} iterations, EV range = [{EV.min():.3f}, {EV.max():.3f}]")
    rng = np.random.default_rng(0)
    panel = bus.simulate_panel(-0.003, -10.0, 0.95, eta,
                               n_buses=30, n_periods=80, rng=rng)
    print(f"Panel shape {panel['x'].shape}, replace freq = {(panel['d'] == 2).mean():.4f}")
    lik = BusEngineLikelihood(panel["x"], panel["d"], M=bus.M)
    prior = MCMCPrior()
    F_oracle = make_F_oracle_exact(bus)
    sampler = BusEngineMCMC(lik, prior, F_oracle)
    chain, diag = sampler.sample(n_iter=400, n_burn=200, rng=rng, verbose=False)
    print(f"Exact‑F MCMC acceptance rates: α={diag['acc_alpha']:.2f}, "
          f"ρ={diag['acc_rho']:.2f}, η={diag['acc_eta']:.2f}")
    print("Self‑check completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="MCMC for bus‑engine DDCM")
    parser.add_argument("--mode", choices=["self", "quick", "standard"], default="self",
                        help="Run mode: 'self' = exact only (default); "
                             "'quick' or 'standard' require PyTorch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "self":
        _run_self_check()
        return

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not installed. Cannot run --mode quick/standard.")
        print("Install torch and scikit-learn, or use --mode self.")
        sys.exit(1)

    # Mode presets (same as original)
    MODE_PRESETS = {
        "quick": dict(
            n_train=200, n_val=80, n_test=80,
            depth=3, width=24, gamma=0.30,
            gl_scale=1e-3, lr=1e-3, batch_size=256,
            epochs=60, patience=15,
            n_buses=50, n_periods=150,
            n_burn=500, n_iter=2000,
        ),
        "standard": dict(
            n_train=2000, n_val=400, n_test=400,
            depth=4, width=32, gamma=0.30,
            gl_scale=1e-3, lr=1e-3, batch_size=512,
            epochs=400, patience=50,
            n_buses=100, n_periods=200,
            n_burn=2000, n_iter=8000,
        ),
    }
    cfg = MODE_PRESETS[args.mode]

    print("=" * 78)
    print("MCMC validation: bus‑engine DDCM (EV‑IID), exact‑F vs KASN‑F")
    print("=" * 78)
    print(f"  mode = {args.mode.upper()}, seed = {args.seed}")
    for k in ("n_train", "depth", "width", "epochs",
              "n_buses", "n_periods", "n_burn", "n_iter"):
        print(f"    {k:<13s} = {cfg[k]}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    bus = BusEngineEVIID(M=90)

    truth = {
        "alpha1": -0.003,
        "alpha2": -10.0,
        "rho": 0.95,
        "eta": np.array([0.36, 0.62, 0.02]),
    }
    print("\n[1/6] Simulating panel under true theta:")
    for k, v in truth.items():
        print(f"        {k} = {v}")

    panel = bus.simulate_panel(
        truth["alpha1"], truth["alpha2"], truth["rho"], truth["eta"],
        n_buses=cfg["n_buses"], n_periods=cfg["n_periods"], rng=rng,
    )
    n_repl = int((panel["d"] == 2).sum())
    print(f"        panel shape = {panel['x'].shape}, "
          f"replacements observed = {n_repl} "
          f"({n_repl / panel['x'].size * 100:.2f}%)")

    print("\n[2/6] Generating KASN training set (independent DP solves)...")
    prior_train = ParameterPriorBox()
    data = make_train_val_test(
        bus, prior_train,
        n_train=cfg["n_train"], n_val=cfg["n_val"], n_test=cfg["n_test"],
        rng=rng, verbose=False,
    )
    print(f"        train={data['X_train'].shape}, "
          f"val={data['X_val'].shape}, test={data['X_test'].shape}")

    print("\n[3/6] Training KASN to approximate F(x; theta)...")
    t0 = time.time()
    model = KASN(
        input_dim=bus.INPUT_DIM, n_samples=cfg["n_train"], gamma=cfg["gamma"],
        kasn_width=cfg["width"], depth=cfg["depth"], dual_reg=False,
        kart_s=4, prune_during_training=False,
    )
    model.fit(data["X_train"], data["y_train"],
              data["X_val"], data["y_val"],
              epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
              patience=cfg["patience"], weight_decay=1e-12,
              l1_reg_scale=0.0, group_lasso_reg_scale=cfg["gl_scale"])
    t_train = time.time() - t0
    F_hat_test = model.predict(data["X_test"])
    test_rmse = float(np.sqrt(np.mean((F_hat_test - data["y_test"]) ** 2)))
    test_r2 = 1.0 - np.sum((F_hat_test - data["y_test"]) ** 2) \
                / np.sum((data["y_test"] - data["y_test"].mean()) ** 2)
    print(f"        KASN training time = {t_train:.1f}s   "
          f"test RMSE = {test_rmse:.4f}, test R² = {test_r2:.4f}")

    lik = BusEngineLikelihood(panel["x"], panel["d"], M=bus.M)
    prior_mcmc = MCMCPrior()
    F_oracle_e = make_F_oracle_exact(bus)
    F_oracle_k = make_F_oracle_kasn(model, M=bus.M)

    print("\n[4/6] MCMC under exact‑F oracle...")
    sampler_e = BusEngineMCMC(lik, prior_mcmc, F_oracle_e)
    t0 = time.time()
    chain_e, diag_e = sampler_e.sample(
        n_iter=cfg["n_iter"], n_burn=cfg["n_burn"], rng=rng, verbose=False)
    t_e = time.time() - t0
    print(f"        time = {t_e:.1f}s   "
          f"acc(α,ρ,η) = ({diag_e['acc_alpha']:.2f}, "
          f"{diag_e['acc_rho']:.2f}, {diag_e['acc_eta']:.2f})")

    print("\n[5/6] MCMC under KASN‑F oracle...")
    sampler_k = BusEngineMCMC(lik, prior_mcmc, F_oracle_k)
    t0 = time.time()
    chain_k, diag_k = sampler_k.sample(
        n_iter=cfg["n_iter"], n_burn=cfg["n_burn"], rng=rng, verbose=False)
    t_k = time.time() - t0
    print(f"        time = {t_k:.1f}s   "
          f"acc(α,ρ,η) = ({diag_k['acc_alpha']:.2f}, "
          f"{diag_k['acc_rho']:.2f}, {diag_k['acc_eta']:.2f})")

    print("\n[6/6] Producing diagnostics...")
    extra = (f"  ($n_{{buses}}{{=}}{cfg['n_buses']}$, "
             f"$T{{=}}{cfg['n_periods']}$, "
             f"replacements${{=}}${n_repl}, KASN $R^2{{=}}{test_r2:.3f}$)")
    _plot_posteriors(chain_e, chain_k, truth,
                     out_path="kasn_mcmc_validation_posteriors.png",
                     title_extra=extra)
    _plot_traces(chain_e, chain_k, truth,
                 out_path="kasn_mcmc_validation_traces.png")
    _print_summary(chain_e, chain_k, truth)

    print(f"\nTimes: KASN train={t_train:.1f}s  "
          f"exact‑F MCMC={t_e:.1f}s  KASN‑F MCMC={t_k:.1f}s  "
          f"speedup={t_e/t_k:.2f}x")
    print("Saved: kasn_mcmc_validation_posteriors.png, kasn_mcmc_validation_traces.png")

if __name__ == "__main__":
    main()