"""Differential privacy for Timber inference outputs.

Implements the Laplace mechanism and the Gaussian mechanism for adding
calibrated noise to model predictions, enabling epsilon-delta differential
privacy guarantees.

Typical usage::

    import numpy as np
    from timber.privacy.dp import DPConfig, apply_dp_noise

    outputs = np.array([[0.8, 0.2]])  # raw model probabilities
    cfg = DPConfig(mechanism="laplace", epsilon=1.0, sensitivity=1.0)
    noisy = apply_dp_noise(outputs, cfg)

References
----------
- Dwork et al., "Calibrating Noise to Sensitivity in Private Data Analysis" (2006)
- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DPConfig:
    """Configuration for differential privacy noise injection.

    Parameters
    ----------
    mechanism:
        Noise mechanism to use.  ``"laplace"`` provides pure
        ε-DP; ``"gaussian"`` provides (ε, δ)-DP.
    epsilon:
        Privacy budget ε > 0.  Smaller values mean stronger privacy
        but more noise.
    delta:
        Relaxation parameter δ for the Gaussian mechanism (unused for
        Laplace).  Typically a small value such as ``1e-5``.
    sensitivity:
        Global L1 sensitivity of the query function (for Laplace) or
        L2 sensitivity (for Gaussian).  For probability outputs bounded
        in [0, 1], sensitivity is at most ``1.0``.
    clip_outputs:
        If True, clip noisy outputs to ``[output_min, output_max]``
        after adding noise.  Useful for probabilities.
    output_min:
        Minimum valid output value (used when ``clip_outputs=True``).
    output_max:
        Maximum valid output value (used when ``clip_outputs=True``).
    seed:
        Optional random seed for reproducibility in tests.
    """
    mechanism: str = "laplace"    # "laplace" | "gaussian"
    epsilon: float = 1.0
    delta: float = 1e-5           # used only by gaussian
    sensitivity: float = 1.0
    clip_outputs: bool = True
    output_min: float = 0.0
    output_max: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.sensitivity <= 0.0:
            raise ValueError(f"sensitivity must be > 0, got {self.sensitivity}")
        if self.mechanism == "gaussian":
            if self.delta <= 0.0 or self.delta >= 1.0:
                raise ValueError(f"delta must be in (0, 1) for gaussian, got {self.delta}")
        if self.mechanism not in ("laplace", "gaussian"):
            raise ValueError(f"Unknown mechanism '{self.mechanism}'. Use 'laplace' or 'gaussian'.")

    @property
    def laplace_scale(self) -> float:
        """b = Δf / ε  (Laplace noise scale)."""
        return self.sensitivity / self.epsilon

    @property
    def gaussian_sigma(self) -> float:
        """σ = Δf · sqrt(2 ln(1.25/δ)) / ε  (Gaussian noise std-dev)."""
        return (
            self.sensitivity
            * math.sqrt(2.0 * math.log(1.25 / self.delta))
            / self.epsilon
        )


@dataclass
class DPReport:
    """Summary of a DP noise-addition operation."""
    mechanism: str
    epsilon: float
    delta: float
    sensitivity: float
    noise_scale: float      # b (Laplace) or σ (Gaussian)
    n_outputs_noised: int
    actual_noise_l2: float  # Euclidean norm of added noise vector

    def summary(self) -> str:
        lines = [
            "Differential Privacy Noise Report",
            f"  Mechanism:    {self.mechanism}",
            f"  ε (epsilon):  {self.epsilon}",
        ]
        if self.mechanism == "gaussian":
            lines.append(f"  δ (delta):    {self.delta:.2e}")
        lines += [
            f"  Sensitivity:  {self.sensitivity}",
            f"  Noise scale:  {self.noise_scale:.6f}",
            f"  Outputs:      {self.n_outputs_noised}",
            f"  ‖noise‖₂:    {self.actual_noise_l2:.6f}",
        ]
        return "\n".join(lines)


def apply_dp_noise(
    outputs: np.ndarray,
    config: DPConfig,
) -> tuple[np.ndarray, DPReport]:
    """Add calibrated DP noise to model output scores.

    Parameters
    ----------
    outputs:
        Array of shape ``(n_samples, n_outputs)`` or ``(n_outputs,)``.
    config:
        DP configuration controlling mechanism, epsilon, delta, etc.

    Returns
    -------
    noisy_outputs:
        Outputs with added noise, optionally clipped to
        ``[config.output_min, config.output_max]``.
    report:
        A ``DPReport`` describing the noise parameters and magnitude.

    Notes
    -----
    The Laplace mechanism guarantees ε-differential privacy.
    The Gaussian mechanism guarantees (ε, δ)-differential privacy.
    """
    rng = np.random.default_rng(config.seed)
    outputs = np.asarray(outputs)
    original_dtype = outputs.dtype
    original_shape = outputs.shape
    outputs = outputs.astype(np.float64)

    flat = outputs.ravel()

    if config.mechanism == "laplace":
        scale = config.laplace_scale
        noise = rng.laplace(loc=0.0, scale=scale, size=flat.shape)
    else:
        sigma = config.gaussian_sigma
        scale = sigma
        noise = rng.normal(loc=0.0, scale=sigma, size=flat.shape)

    noisy_flat = flat + noise
    noise_l2 = float(np.linalg.norm(noise))

    noisy = noisy_flat.reshape(original_shape)

    if config.clip_outputs:
        noisy = np.clip(noisy, config.output_min, config.output_max)

    report = DPReport(
        mechanism=config.mechanism,
        epsilon=config.epsilon,
        delta=config.delta if config.mechanism == "gaussian" else 0.0,
        sensitivity=config.sensitivity,
        noise_scale=scale,
        n_outputs_noised=int(flat.size),
        actual_noise_l2=noise_l2,
    )

    return noisy.astype(original_dtype), report


def calibrate_epsilon(
    target_noise_std: float,
    sensitivity: float,
    mechanism: str = "laplace",
    delta: float = 1e-5,
) -> float:
    """Compute the epsilon value that achieves a target noise standard deviation.

    Parameters
    ----------
    target_noise_std:
        Desired standard deviation of the added noise.
    sensitivity:
        Global sensitivity of the query.
    mechanism:
        ``"laplace"`` or ``"gaussian"``.
    delta:
        Used only for the Gaussian mechanism.

    Returns
    -------
    epsilon:
        The privacy budget ε that results in the desired noise level.
    """
    if mechanism == "laplace":
        # std of Laplace(0, b) = b * sqrt(2); b = Δf / ε
        # target_noise_std = (Δf / ε) * sqrt(2)
        # ε = Δf * sqrt(2) / target_noise_std
        return sensitivity * math.sqrt(2.0) / target_noise_std
    else:
        # std of Gaussian = σ; σ = Δf * sqrt(2 ln(1.25/δ)) / ε
        # ε = Δf * sqrt(2 ln(1.25/δ)) / target_noise_std
        return (
            sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / target_noise_std
        )
