"""
Self-contained gnorm-driven LR scheduler for SpicyDion.

Embeds a causal gnorm variance tracker that emits two signals:
  1. "stable"     — initial transient is over (MA variance < stable_thresh).
                    Triggers cold → ramp → cruise transition.
  2. "tapped_out" — model has extracted what it can at current LR
                    (MA variance < tap_thresh for tap_confirm consecutive steps).
                    Each tap triggers a compounding 5% LR step-down.

Phases (cold → ramp → cruise, with step-downs during cruise):
  Cold:   base_lr / 10, adaptive inactive. Waiting for stable signal.
  Ramp:   linear from base_lr/10 → base_lr over ramp_steps. Adaptive active.
  Cruise: scheduled_lr = base_lr * (smooth_gnorm / smooth_max)^power.
          When tapped_out fires, base_lr *= (1 - step_down_fraction).
          The LR cut causes a gnorm transient, resetting the tap detector.
          When it re-settles, step down again. Repeat until lr_floor.

No cosine decay. No total_steps. All dynamics driven by actual gnorm signal.

The scheduler owns group["lr"] and group["scheduled_lr"]. The optimizer
must NOT write to scheduled_lr.
"""

import math
from collections import deque


class GnormScheduler:
    """
    Gnorm-driven LR scheduler for SpicyDion.

    Parameters
    ----------
    base_lr : float
        Target peak learning rate (will be reduced by step-downs).
    n_layers : int
        Model depth. Windows scale as 20 * n_layers to account for
        deeper models having longer gradient transients.

    Tracker params:
    stable_thresh : float
        MA variance below this → initial transient is over.
    tap_thresh : float
        MA variance below this → model is tapped out at current LR.

    Ramp params:
    ramp_steps : int
        Steps to linearly ramp from base_lr/10 → base_lr after stability.

    Cruise params:
    schedule_power : float
        Exponent on gnorm ratio (smooth/max). >1 = more aggressive decay.
    gnorm_smooth_beta : float
        EMA decay for smoothing gnorm during cruise.

    Step-down params:
    step_down_fraction : float
        Fraction to reduce base_lr on each plateau (0.05 = 5%).
    lr_floor : float
        Early exit when scheduled LR drops below this.
    """

    def __init__(
        self,
        base_lr: float,
        n_layers: int = 1,
        # Tracker (base windows scaled by 20 * n_layers)
        stable_thresh: float = 0.02,
        tap_thresh: float = 0.0003,
        # Ramp
        ramp_steps: int = 100,
        # Cruise
        schedule_power: float = 1.0,
        gnorm_smooth_beta: float = 0.99,
        # Step-down
        step_down_fraction: float = 0.05,
        lr_floor: float = 1e-9,
    ):
        self.base_lr = base_lr
        self._original_base_lr = base_lr

        # Window sizes scale with model depth: 20 * L
        _w = 20 * max(1, n_layers)
        ma_window = _w
        var_window = _w
        tap_confirm = _w
        cooloff_steps = 3 * _w

        # --- Tracker state ---
        self._ma_buf: deque[float] = deque(maxlen=ma_window)
        self._ma_sum: float = 0.0
        self._var_buf: deque[float] = deque(maxlen=var_window)
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

        # --- Phase state ---
        self.phase: str = "cold"  # cold → ramp → cruise
        self._ramp_start_step: int = 0
        self.ramp_steps = ramp_steps

        # --- Cruise state ---
        self.schedule_power = schedule_power
        self._gnorm_smooth_beta = gnorm_smooth_beta
        self._smooth_gnorm: float = 0.0
        self._smooth_max_gnorm: float = 0.0
        self._cruise_initialized: bool = False

        # --- Step-down ---
        self._step_down_fraction = step_down_fraction
        self._lr_floor = lr_floor

        # --- Public read-only ---
        self.current_lr: float = base_lr * 0.1
        self.gnorm_ratio: float = 1.0
        self._ramp_progress: float = 0.0

    # ------------------------------------------------------------------
    # Tracker: update MA + variance, return (stable_triggered, tap_triggered)
    # ------------------------------------------------------------------

    def _tracker_step(self, gnorm: float, step_num: int, check_tap: bool = False) -> tuple[bool, bool]:
        """
        Feed one gnorm. Updates MA + variance continuously.
        Returns (stable_just_triggered, tap_just_triggered).
        Tap detection only runs when check_tap=True (cruise phase).
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
            return False, False

        mean = sum(self._var_buf) / len(self._var_buf)
        var = sum((x - mean) ** 2 for x in self._var_buf) / len(self._var_buf)
        self.current_variance = var

        # Check stable (one-shot)
        stable_triggered = False
        if self._stable_since is None and var < self._stable_thresh:
            self._stable_since = step_num
            stable_triggered = True

        # Check tapped out (only during cruise; confirmation + cooloff)
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

    def step(self, gnorm: float, step_num: int) -> dict:
        """
        Feed one gnorm observation. Returns dict:
            lr: float             - scheduled LR for this step
            adaptive_active: bool - per-neuron adaptive LR active?
            phase: str            - "cold", "ramp", or "cruise"
            gnorm_ratio: float    - smooth/max (1.0 if not cruise)
            tap_count: int        - number of plateau step-downs so far
            ma_variance: float|None - current MA variance (for logging)
            early_exit: bool      - True if LR has dropped below floor
        """
        in_cruise = self.phase == "cruise"
        stable_triggered, tap_triggered = self._tracker_step(gnorm, step_num, check_tap=in_cruise)

        # --- Phase transitions ---
        if self.phase == "cold":
            if stable_triggered:
                self.phase = "ramp"
                self._ramp_start_step = step_num
            else:
                self.current_lr = self.base_lr * 0.1
                self._ramp_progress = 0.0
                return self._result(adaptive_active=False)

        if self.phase == "ramp":
            t = (step_num - self._ramp_start_step) / max(1, self.ramp_steps)
            if t >= 1.0:
                self.phase = "cruise"
                t = 1.0
            self._ramp_progress = t
            self.current_lr = self.base_lr * (0.1 + 0.9 * t)
            return self._result(adaptive_active=True)

        # --- Cruise phase ---
        assert self.phase == "cruise"
        self._ramp_progress = 1.0

        # Plateau step-down: reduce base_lr
        if tap_triggered:
            self.base_lr *= (1.0 - self._step_down_fraction)

        # Gnorm ratio tracking
        b = self._gnorm_smooth_beta
        if not self._cruise_initialized:
            self._smooth_gnorm = gnorm
            self._smooth_max_gnorm = gnorm
            self._cruise_initialized = True
        else:
            self._smooth_gnorm = b * self._smooth_gnorm + (1.0 - b) * gnorm
            self._smooth_max_gnorm = max(
                self._smooth_max_gnorm, self._smooth_gnorm
            )

        self.gnorm_ratio = self._smooth_gnorm / max(
            self._smooth_max_gnorm, 1e-12
        )
        # Pre-map ratio with exponent, then apply cosine S-curve (AutoNorMuon style).
        # ratio=1 → cos(0)=1 → mult=1.0 (full LR)
        # ratio=0 → cos(π)=-1 → mult=0.0 (zero LR)
        # Cosine is gentler near extremes, steeper in the middle.
        mapped = self.gnorm_ratio ** self.schedule_power
        mult = 0.5 * (1.0 + math.cos(math.pi * (1.0 - mapped)))
        self.current_lr = self.base_lr * mult

        return self._result(adaptive_active=True)

    def _result(self, adaptive_active: bool) -> dict:
        return {
            "lr": self.current_lr,
            "adaptive_active": adaptive_active,
            "phase": self.phase,
            "gnorm_ratio": self.gnorm_ratio,
            "tap_count": self.tap_count,
            "ma_variance": self.current_variance,
            "early_exit": self.current_lr < self._lr_floor,
            "ramp_progress": self._ramp_progress,
        }
