"""
Self-contained gnorm-driven LR scheduler.

Two mechanisms drive LR downward (never upward):

  1. CV deviation: coefficient of variation (std/mean) of the gnorm MA
     window measures how unstable gradients are. During cruise, LR is
     scaled down by CV: lr = base_lr * (1 - cv). Higher CV = more
     instability = more aggressive LR reduction. CV naturally decreases
     as model converges (gnorms stabilize), so LR trends monotonically
     downward without explicit scheduling.

  2. Tap detection: when gnorm MA variance flatlines (model stuck in
     local minimum, no CV signal), compounding 5% step-down to base_lr
     kicks the model out. Cooloff period prevents rapid-fire taps.

Per-neuron adaptive LR (in the optimizer) is bidirectional — individual
neurons can have their LR go up or down based on their own ratio.

Phases:
  Cold:   base_lr / 10, adaptive inactive. Waiting for stable gnorm signal.
  Ramp:   linear from base_lr/10 → base_lr over ramp_steps. Adaptive inactive.
  Settle: full base_lr for 1x window. Adaptive inactive. Builds CV baseline.
  Cruise: CV-driven LR reduction + tap detection. Adaptive active.

No cosine decay. No total_steps. Dynamics driven by actual gnorm signals.

The scheduler owns group["lr"] and group["scheduled_lr"]. The optimizer
must NOT write to scheduled_lr.
"""

import math
from collections import deque


class GnormScheduler:
    """
    Gnorm CV + tap-detection LR scheduler.

    Parameters
    ----------
    base_lr : float
        Target peak learning rate (only decreases from here).
    n_layers : int
        Model depth. Windows scale as 20 * n_layers.

    Stability params:
    stable_thresh : float
        Gnorm MA variance below this → initial transient is over.
    tap_thresh : float
        Gnorm MA variance below this → model is tapped out at current LR.

    Ramp params:
    ramp_steps : int
        Steps to linearly ramp from base_lr/10 → base_lr after stability.

    Step-down params:
    step_down_fraction : float
        Fraction to reduce base_lr on each tap plateau (0.05 = 5%).
    lr_floor : float
        Early exit when scheduled LR drops below this.
    """

    def __init__(
        self,
        base_lr: float,
        n_layers: int = 1,
        # Stability tracker (window = 20 * n_layers)
        stable_thresh: float = 0.01,
        tap_thresh: float = 0.0003,
        # Ramp
        ramp_steps: int = 100,
        # Step-down
        step_down_fraction: float = 0.05,
        lr_floor: float = 1e-9,
        # Legacy (accepted but ignored for backward compat)
        schedule_power: float = 1.0,
        loss_smooth_beta: float = 0.99,
        fast_beta: float = 0.95,
        slow_beta: float = 0.999,
    ):
        self.base_lr = base_lr
        self._original_base_lr = base_lr

        # Window sizes scale with model depth: 20 * L
        _w = 20 * max(1, n_layers)
        tap_confirm = _w
        cooloff_steps = 3 * _w

        # --- MA + variance tracker ---
        self._ma_buf: deque[float] = deque(maxlen=_w)
        self._ma_sum: float = 0.0
        self._var_buf: deque[float] = deque(maxlen=_w)
        self._stable_thresh = stable_thresh
        self._tap_thresh = tap_thresh
        self._tap_confirm = tap_confirm
        self._cooloff_steps = cooloff_steps

        self._stable_since: int | None = None
        self._tap_run: int = 0
        self._cooloff_remaining: int = 0
        self.tap_count: int = 0

        # Current tracker outputs (for logging)
        self.current_ma: float = 0.0
        self.current_variance: float | None = None
        self.current_cv: float = 0.0  # coefficient of variation: std/mean

        # --- Phase state ---
        self.phase: str = "cold"  # cold → ramp → cruise (settle is sub-phase of cruise)
        self._ramp_start_step: int = 0
        self.ramp_steps = ramp_steps
        self._cruise_settle_remaining: int = _w  # 1x window settle before adaptive activates

        # --- Step-down ---
        self._step_down_fraction = step_down_fraction
        self._lr_floor = lr_floor

        # --- Public read-only ---
        self.current_lr: float = base_lr * 0.1
        self.deviation: float = 0.0  # = CV during cruise, 0 otherwise
        self.loss_ratio: float = 1.0  # backward compat: 1 - deviation
        self._ramp_progress: float = 0.0

    # ------------------------------------------------------------------
    # MA + variance + CV tracker
    # ------------------------------------------------------------------

    def _tracker_step(self, gnorm: float, step_num: int, check_tap: bool = False) -> tuple[bool, bool]:
        """
        Feed one gnorm. Updates MA, variance, and CV continuously.
        Returns (stable_just_triggered, tap_just_triggered).
        """
        # Update causal MA
        if len(self._ma_buf) == self._ma_buf.maxlen:
            self._ma_sum -= self._ma_buf[0]
        self._ma_buf.append(gnorm)
        self._ma_sum += gnorm
        self.current_ma = self._ma_sum / len(self._ma_buf)

        # Update variance of MA
        self._var_buf.append(self.current_ma)
        if len(self._var_buf) < self._var_buf.maxlen:
            self.current_variance = None
            self.current_cv = 0.0
            return False, False

        mean = sum(self._var_buf) / len(self._var_buf)
        var = sum((x - mean) ** 2 for x in self._var_buf) / len(self._var_buf)
        self.current_variance = var

        # CV = std / mean (coefficient of variation)
        if mean > 0:
            self.current_cv = math.sqrt(var) / mean
        else:
            self.current_cv = 0.0

        # Check stable (one-shot)
        stable_triggered = False
        if self._stable_since is None and var < self._stable_thresh:
            self._stable_since = step_num
            stable_triggered = True

        # Check tapped out (only during settled cruise; confirmation + cooloff)
        tap_triggered = False
        if not check_tap:
            self._tap_run = 0
        elif self._cooloff_remaining > 0:
            self._cooloff_remaining -= 1
            self._tap_run = 0
        elif var < self._tap_thresh:
            self._tap_run += 1
            if self._tap_run >= self._tap_confirm:
                self.tap_count += 1
                tap_triggered = True
                self._tap_run = 0
                self._cooloff_remaining = self._cooloff_steps
        else:
            self._tap_run = 0

        return stable_triggered, tap_triggered

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, gnorm: float, loss: float, step_num: int) -> dict:
        """
        Feed one gnorm observation. Loss is accepted for API compat but
        not used for LR control (gnorm CV drives everything).

        Returns dict:
            lr: float             - scheduled LR for this step
            adaptive_active: bool - per-neuron adaptive LR active?
            phase: str            - "cold", "ramp", "settle", or "cruise"
            deviation: float      - CV (std/mean) of gnorm MA window
            loss_ratio: float     - backward compat (= 1 - deviation, clamped)
            tap_count: int        - number of plateau step-downs so far
            ma_variance: float|None - current gnorm MA variance (for logging)
            early_exit: bool      - True if LR has dropped below floor
            ramp_progress: float  - 0→1 during ramp, 1.0 after
        """
        in_cruise = self.phase == "cruise"
        _settled = in_cruise and self._cruise_settle_remaining <= 0
        stable_triggered, tap_triggered = self._tracker_step(gnorm, step_num, check_tap=_settled)

        # --- Phase transitions ---
        if self.phase == "cold":
            if stable_triggered:
                self.phase = "ramp"
                self._ramp_start_step = step_num
            else:
                self.current_lr = self.base_lr * 0.1
                self._ramp_progress = 0.0
                return self._result(adaptive_active=False, phase_display="cold")

        if self.phase == "ramp":
            t = (step_num - self._ramp_start_step) / max(1, self.ramp_steps)
            if t >= 1.0:
                self.phase = "cruise"
                t = 1.0
            self._ramp_progress = t
            self.current_lr = self.base_lr * (0.1 + 0.9 * t)
            return self._result(adaptive_active=False, phase_display="ramp")

        # --- Cruise phase (includes settle sub-phase) ---
        assert self.phase == "cruise"
        self._ramp_progress = 1.0

        # Settle delay: 1x window after ramp ends before activating adaptive + CV.
        if self._cruise_settle_remaining > 0:
            self._cruise_settle_remaining -= 1
            self.current_lr = self.base_lr
            return self._result(adaptive_active=False, phase_display="settle")

        # --- Active cruise: CV-driven step-down + tap detection ---

        # Tap step-down: compounding 5% when variance flatlines
        if tap_triggered:
            self.base_lr *= (1.0 - self._step_down_fraction)

        # CV-driven LR: lr = base_lr * (1 - cv), clamped to [0, base_lr].
        # CV is always >= 0. As model converges, gnorms stabilize, CV → 0,
        # so LR approaches base_lr. When gnorms spike, CV rises, LR drops.
        cv = self.current_cv
        self.deviation = cv
        self.current_lr = self.base_lr * max(0.0, 1.0 - cv)

        # Backward compat
        self.loss_ratio = max(0.0, 1.0 - cv)

        return self._result(adaptive_active=True, phase_display="cruise")

    def _result(self, adaptive_active: bool, phase_display: str) -> dict:
        return {
            "lr": self.current_lr,
            "adaptive_active": adaptive_active,
            "phase": phase_display,
            "deviation": self.deviation,
            "loss_ratio": self.loss_ratio,
            "tap_count": self.tap_count,
            "ma_variance": self.current_variance,
            "early_exit": self.current_lr < self._lr_floor,
            "ramp_progress": self._ramp_progress,
        }
