from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import optimize, stats
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class MIMICResult:
    """Container for fitted MIMIC model results."""
    causes: list[str]
    indicators: list[str]
    n_obs: int
    gamma: np.ndarray
    lambda_: np.ndarray
    psi: float
    theta: np.ndarray
    phi: np.ndarray
    gamma_se: np.ndarray
    lambda_se: np.ndarray
    chi2: float
    df: int
    pvalue: float
    rmsea: float
    cfi: float
    tli: float
    aic: float
    bic: float
    loglik: float
    converged: bool
    fit_function: float = 0.0
    sample_cov: np.ndarray = field(default=None, repr=False)
    implied_cov: np.ndarray = field(default=None, repr=False)

    def gamma_z(self):
        return self.gamma / self.gamma_se

    def gamma_p(self):
        return 2 * (1 - stats.norm.cdf(np.abs(self.gamma_z())))

    def lambda_z(self):
        z = np.full_like(self.lambda_, np.nan, dtype=float)
        for i in range(1, len(self.lambda_)):
            if self.lambda_se[i] > 0:
                z[i] = self.lambda_[i] / self.lambda_se[i]
        return z

    def lambda_p(self):
        z = self.lambda_z()
        return 2 * (1 - stats.norm.cdf(np.abs(z)))

    def n_significant_causes(self, alpha=0.10):
        return int(np.sum(self.gamma_p() < alpha))

    def n_significant_indicators(self, alpha=0.10):
        p = self.lambda_p()
        return int(np.sum(p[1:] < alpha))


def _build_sigma(gamma, lam, psi, theta, phi):
    """Implied population covariance of (y, x)."""
    p = len(lam)
    q = len(gamma)
    var_eta = float(gamma @ phi @ gamma + psi)
    Sigma_yy = np.outer(lam, lam) * var_eta + np.diag(theta)
    Sigma_yx = np.outer(lam, phi @ gamma)
    Sigma_xx = phi
    Sigma = np.zeros((p + q, p + q))
    Sigma[:p, :p] = Sigma_yy
    Sigma[:p, p:] = Sigma_yx
    Sigma[p:, :p] = Sigma_yx.T
    Sigma[p:, p:] = Sigma_xx
    return Sigma


def _ml_fit_function(Sigma, S):
    sign_S, logdet_S = np.linalg.slogdet(S)
    sign_Si, logdet_Si = np.linalg.slogdet(Sigma)
    if sign_Si <= 0 or sign_S <= 0:
        return 1e10
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return 1e10
    k = S.shape[0]
    return logdet_Si + np.trace(S @ Sigma_inv) - logdet_S - k


def _pack(gamma, lam_free, log_psi, log_theta):
    """Pack parameters for the optimiser. We optimise log of variances
    to keep them positive. lambda_1 is fixed at 1 (not in the vector)."""
    return np.concatenate([gamma, lam_free, [log_psi], log_theta])


def _unpack(params, q, p):
    gamma = params[:q]
    lam_free = params[q:q + (p - 1)]
    log_psi = params[q + (p - 1)]
    log_theta = params[q + (p - 1) + 1: q + (p - 1) + 1 + p]
    lam = np.concatenate([[1.0], lam_free])
    psi = float(np.exp(log_psi))
    theta = np.exp(log_theta)
    return gamma, lam, psi, theta


def fit_mimic(data: pd.DataFrame,
              causes: Sequence[str],
              indicators: Sequence[str],
              standardize: bool = True,
              n_starts: int = 5,
              maxiter: int = 1000,
              random_state: int = 0) -> MIMICResult:
    """
    Fit a 1-factor MIMIC model by Maximum Likelihood.

    The latent variable is identified by setting the loading of the
    first indicator to 1.0 (Schneider convention).

    Parameters
    ----------
    data        : DataFrame with all variables, no NaN rows used for fitting.
    causes      : list of cause column names.
    indicators  : list of indicator column names. The first one is the
                  reference (its loading is fixed to 1).
    standardize : if True, all variables are z-scored before fit (this
                  yields a 'standardised' MIMIC, common in the literature
                  and helps numerical stability).

    Returns
    -------
    MIMICResult with parameter estimates, SEs, fit indices.
    """
    rng = np.random.default_rng(random_state)
    causes = list(causes); indicators = list(indicators)

    cols = indicators + causes
    df = data[cols].dropna().copy()
    n = len(df)
    if n < len(cols) + 2:
        raise ValueError(f"Too few observations ({n}) for {len(cols)} variables.")

    if standardize:
        df = (df - df.mean()) / df.std(ddof=1)

    p = len(indicators); q = len(causes)
    Y = df[indicators].values
    X = df[causes].values

    Z = np.column_stack([Y, X])
    Z = Z - Z.mean(axis=0)
    S = (Z.T @ Z) / n
    phi = S[p:, p:]

    def neg_loglik(params):
        gamma, lam, psi, theta = _unpack(params, q, p)
        Sigma = _build_sigma(gamma, lam, psi, theta, phi)
        return _ml_fit_function(Sigma, S)

    best = None
    for start in range(n_starts):
        if start == 0:
            y_avg = Y.mean(axis=1)
            try:
                gamma0 = np.linalg.lstsq(X, y_avg, rcond=None)[0]
            except Exception:
                gamma0 = np.zeros(q)
            lam_free0 = np.ones(p - 1)
            log_psi0 = np.log(0.5)
            log_theta0 = np.log(np.maximum(np.diag(S)[:p] * 0.5, 1e-3))
        else:
            gamma0 = rng.normal(0, 0.3, q)
            lam_free0 = rng.normal(1, 0.3, p - 1)
            log_psi0 = np.log(rng.uniform(0.2, 0.8))
            log_theta0 = np.log(rng.uniform(0.2, 0.8, p))

        params0 = _pack(gamma0, lam_free0, log_psi0, log_theta0)
        try:
            res = optimize.minimize(neg_loglik, params0, method='BFGS',
                                    options={'maxiter': maxiter, 'gtol': 1e-6})
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("Optimisation failed for all starting values.")

    gamma_hat, lam_hat, psi_hat, theta_hat = _unpack(best.x, q, p)
    Sigma_hat = _build_sigma(gamma_hat, lam_hat, psi_hat, theta_hat, phi)

    npar = len(best.x)
    try:
        if hasattr(best, 'hess_inv') and best.hess_inv is not None:
            Hinv = np.asarray(best.hess_inv)
            cov = (2.0 / n) * Hinv
            diag_cov = np.diag(cov)
            ses_packed = np.sqrt(np.maximum(diag_cov, 1e-12))
        else:
            ses_packed = np.full(npar, np.nan)
    except Exception:
        ses_packed = np.full(npar, np.nan)

    gamma_se = ses_packed[:q]
    lam_free_se = ses_packed[q:q + (p - 1)]
    lam_se = np.concatenate([[0.0], lam_free_se])

    F_ML = best.fun
    chi2 = n * F_ML
    k = p + q
    n_distinct = k * (k + 1) / 2
    n_free = q + (p - 1) + 1 + p + q * (q + 1) / 2 
    
    df_model = n_distinct - n_free
    df_model = max(int(round(df_model)), 1)

    pvalue = 1 - stats.chi2.cdf(chi2, df_model) if chi2 > 0 else 1.0

    diag_S = np.diag(np.diag(S))
    F_base = (np.log(np.linalg.det(diag_S))
              + np.trace(S @ np.linalg.inv(diag_S))
              - np.log(np.linalg.det(S)) - k)
    chi2_base = n * F_base
    df_base = k * (k - 1) / 2

    a = max(chi2 - df_model, 0)
    b = max(chi2_base - df_base, 0)
    cfi = 1 - a / b if b > 0 else np.nan

    if df_base > 0 and df_model > 0:
        tli_num = chi2_base / df_base - chi2 / df_model
        tli_den = chi2_base / df_base - 1
        tli = tli_num / tli_den if tli_den != 0 else np.nan
    else:
        tli = np.nan

    rmsea_num = max(chi2 - df_model, 0)
    rmsea = np.sqrt(rmsea_num / (df_model * n)) if df_model > 0 and n > 0 else np.nan

    loglik = -0.5 * n * (k * np.log(2 * np.pi) + np.linalg.slogdet(Sigma_hat)[1]
                        + np.trace(S @ np.linalg.inv(Sigma_hat)))
    npar_aic = q + (p - 1) + 1 + p
    aic = -2 * loglik + 2 * npar_aic
    bic = -2 * loglik + npar_aic * np.log(n)

    return MIMICResult(
        causes=causes, indicators=indicators, n_obs=n,
        gamma=gamma_hat, lambda_=lam_hat, psi=psi_hat, theta=theta_hat,
        phi=phi,
        gamma_se=gamma_se, lambda_se=lam_se,
        chi2=chi2, df=df_model, pvalue=pvalue,
        rmsea=rmsea, cfi=cfi, tli=tli,
        aic=aic, bic=bic, loglik=loglik,
        converged=best.success,
        fit_function=F_ML,
        sample_cov=S, implied_cov=Sigma_hat,
    )


def predict_latent(result: MIMICResult, data: pd.DataFrame, standardize: bool = True) -> np.ndarray:
    """
    Bartlett-style factor score for the latent variable using the
    structural part: eta_hat = gamma' x_t (the structural prediction).
    Returns z-scored series.
    """
    df = data[result.causes].copy()
    if standardize:
        df = (df - df.mean()) / df.std(ddof=1)
    eta = df.values @ result.gamma
    return eta