"""Confidence signals and their calibration. §4.5, §8.5, §13.3.

The calibrated generative probability drives routing only (widen the beam? invoke the
lexical channel?). Final ranking uses the reranker score, calibrated separately with
isotonic regression. A confidence that does not correlate with correctness cannot drive
the fallback policy, and an uncalibrated trigger is how the lexical channel silently eats
the traffic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MARGIN_CAP = 99.0  # a node with one child has no runner-up; treat as certain


def entropy(logp: np.ndarray) -> float:
    p = np.exp(logp)
    ok = p > 0
    return float(-(p[ok] * logp[ok]).sum())


def margin(logp: np.ndarray) -> float:
    """``log p(top) - log p(runner-up)`` over the valid children of one node."""
    finite = np.sort(logp[np.isfinite(logp)])[::-1]
    return float(finite[0] - finite[1]) if len(finite) > 1 else MARGIN_CAP


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))  # no overflow


def _nll(w, X, y):
    z = X @ w
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


@dataclass(frozen=True)
class Calibrator:
    temperature: float = 1.0
    bias: float = 0.0

    def prob(self, s_top: float) -> float:
        return float(_sigmoid(s_top / self.temperature + self.bias))


def fit_temperature(scores: np.ndarray, labels: np.ndarray, iters: int = 100) -> Calibrator:
    """``T*, b* = argmin -sum y log sigma(s/T + b) + (1-y) log(1 - sigma(s/T + b))``.

    Damped Newton on the two-parameter logistic, from zero, halving any step that raises
    the loss. Raises if the fitted slope is not positive: that means s_top does not track
    relevance and stage D must fail loudly.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    X = np.stack([s, np.ones_like(s)], 1)
    w = np.zeros(2)
    loss = _nll(w, X, y)
    for _ in range(iters):
        p = _sigmoid(X @ w)
        g = X.T @ (p - y)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        t = 1.0
        while _nll(w - t * step, X, y) > loss and t > 1e-8:
            t *= 0.5
        w = w - t * step
        new = _nll(w, X, y)
        if loss - new < 1e-12:
            break
        loss = new
    if w[0] <= 0:
        raise ValueError("sequence score is not positively related to relevance; cannot calibrate")
    return Calibrator(temperature=float(1.0 / w[0]), bias=float(w[1]))


@dataclass(frozen=True)
class Isotonic:
    """Monotone score -> probability map, pool-adjacent-violators, linear between blocks."""

    xs: tuple[float, ...]
    ys: tuple[float, ...]

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> Isotonic:
        order = np.argsort(x, kind="stable")
        x = np.asarray(x, dtype=np.float64)[order]
        y = np.asarray(y, dtype=np.float64)[order]
        vals, wts, xsum = [], [], []
        for xi, yi in zip(x, y, strict=True):
            vals.append(yi)
            wts.append(1.0)
            xsum.append(xi)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]
                vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
                xsum[-2] += xsum[-1]
                wts[-2] = w
                del vals[-1], wts[-1], xsum[-1]
        return cls(tuple(xs / w for xs, w in zip(xsum, wts, strict=True)), tuple(vals))

    def __call__(self, x):
        return np.interp(x, self.xs, self.ys)
