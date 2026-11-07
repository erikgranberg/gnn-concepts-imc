"""
Survival time simulation from a Cox PH model.

Survival times are drawn from an exponential distribution with hazard:

    λ_i = exp(β · signal_i + σ · ε_i)

where:
  - signal_i  is the spatial feature driving survival (regime-specific)
  - β         is the effect size (strength of the spatial signal)
  - σ         is the noise level (difficulty knob)
  - ε_i       ~ N(0, 1)

A higher signal → lower hazard → longer survival (the spatial feature is
always coded so that signal=1 is the "good" end and signal=0 is "bad").

Censoring is applied independently via an exponential censoring process
with rate λ_c chosen to yield approximately the target event rate.
"""

import numpy as np


def simulate_survival(
    signals: np.ndarray,
    beta: float = 2.0,
    noise: float = 0.5,
    censoring_rate: float = 0.4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate censored survival times from a Cox exponential model.

    Args:
        signals:        Array of per-patient spatial signals in [0, 1].
                        Higher signal = better prognosis (lower hazard).
        beta:           Effect size. Larger β = stronger spatial signal.
        noise:          Noise level σ. Larger σ = harder task.
        censoring_rate: Approximate proportion of censored observations.
        rng:            Random number generator.

    Returns:
        times:  Observed times (months, scaled to realistic range).
        events: Binary event indicators (1 = event, 0 = censored).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signals)
    epsilon = rng.standard_normal(n)

    # Log-hazard: high signal → negative log-hazard → longer survival.
    # Centred on its own population mean so the absolute survival-time scale
    # (set by `scale` below) stays ~40-80 months regardless of β or how the
    # per-regime signal happens to be distributed over [0, 1]. Without this,
    # -beta*signals shifts the whole population's hazard down whenever mean
    # signal > 0, which blows up event_times exponentially (e.g. mean signal
    # 0.5 at beta=2 alone inflates the mean survival time ~2.7x). Centring is
    # a free normalisation: shifting log-hazard by a constant is equivalent to
    # rescaling the (unspecified) baseline hazard, so it leaves every
    # patient's relative risk ordering — and therefore C-index — unchanged.
    log_hazard = -beta * signals + noise * epsilon
    log_hazard = log_hazard - log_hazard.mean()
    hazard = np.exp(log_hazard)

    # Event times from Exponential(hazard), scaled to months.
    # Scale factor chosen so median survival ~ 40-80 months (median of
    # Exponential(scale) is scale * ln(2) ≈ 42 months at the population's
    # average hazard of ~1 after centring above).
    scale = 60.0
    event_times = rng.exponential(scale / hazard)

    # Censoring times from Exponential(rate chosen to hit target censoring)
    # P(censored) ≈ censoring_rate when λ_c ≈ (1-censoring_rate)/censoring_rate * hazard.
    # Uses the *median* hazard as the representative value, not the mean: after
    # centring, median(hazard) ≈ 1 robustly regardless of β/noise, while
    # mean(hazard) inflates with them (Jensen's inequality on the lognormal-like
    # hazard), which previously pushed observed censoring well above the target
    # (e.g. ~63% instead of 40% at high β/noise).
    rep_hazard = np.median(hazard)
    lambda_c = (censoring_rate / (1.0 - censoring_rate)) * (rep_hazard / scale)
    censor_times = rng.exponential(1.0 / lambda_c, size=n)

    times  = np.minimum(event_times, censor_times)
    events = (event_times <= censor_times).astype(int)

    return times, events
