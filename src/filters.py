"""Filters for cursor smoothing and jitter suppression."""
import math
from typing import Optional, Tuple


class LowPassFilter:
    """First-order low-pass filter with exponential smoothing."""

    def __init__(self, alpha: float = 0.5):
        self._alpha = max(0.0, min(1.0, alpha))
        self._s: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self._alpha = max(0.0, min(1.0, alpha))
        if self._s is None:
            self._s = value
        else:
            self._s = self._alpha * value + (1.0 - self._alpha) * self._s
        return self._s

    def last_value(self) -> Optional[float]:
        return self._s

    def reset(self) -> None:
        self._s = None


class OneEuroFilter:
    """1-Euro Filter for 1D signals.

    An adaptive low-pass filter providing high stability at low speeds
    and low latency (minimal lag) during rapid movement.
    Reference: Casiez, Roussel, Vogel (CHI 2012).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: Optional[float] = None

    def _calc_alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            # Fallback if no timestamp provided (assumes 60Hz nominal ~0.016s)
            dt = 0.016
        elif self._last_time is None:
            dt = 0.016
            self._last_time = timestamp
        else:
            raw_dt = timestamp - self._last_time
            if raw_dt <= 0:
                dt = 0.016
            elif raw_dt > 0.5:
                # Long pause or gap between frames: reset velocity history so derivative doesn't glitch
                self._dx_filter.reset()
                dt = 0.033
                self._last_time = timestamp
            else:
                dt = max(0.003, raw_dt)
                self._last_time = timestamp

        # Estimate rate of change (derivative)
        prev_x = self._x_filter.last_value()
        dx = 0.0 if prev_x is None else (x - prev_x) / dt
        d_alpha = self._calc_alpha(self.d_cutoff, dt)
        filtered_dx = self._dx_filter.filter(dx, d_alpha)

        # Dynamic cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(filtered_dx)
        alpha = self._calc_alpha(cutoff, dt)

        return self._x_filter.filter(x, alpha)

    def reset(self) -> None:
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None


class PointOneEuroFilter:
    """2D One-Euro Filter for coordinate pairs (X, Y)."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self._filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def update_params(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._filter_x.min_cutoff = float(min_cutoff)
        self._filter_x.beta = float(beta)
        self._filter_x.d_cutoff = float(d_cutoff)

        self._filter_y.min_cutoff = float(min_cutoff)
        self._filter_y.beta = float(beta)
        self._filter_y.d_cutoff = float(d_cutoff)

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        fx = self._filter_x.filter(x, timestamp)
        fy = self._filter_y.filter(y, timestamp)
        return fx, fy

    def reset(self) -> None:
        self._filter_x.reset()
        self._filter_y.reset()


class DeadZoneFilter:
    """Deadzone filter around a neutral point to eliminate resting drift.

    Provides continuous transition at the deadzone edge to avoid sudden jumps.
    """

    def __init__(self, radius: float = 0.02):
        self.radius = max(0.0, float(radius))

    def apply(
        self,
        x: float,
        y: float,
        neutral_x: float,
        neutral_y: float,
    ) -> Tuple[float, float, float]:
        """Calculates deadzone-filtered offset from neutral point.

        Returns:
            (filtered_dx, filtered_dy, raw_distance)
        """
        dx = x - neutral_x
        dy = y - neutral_y
        dist = math.hypot(dx, dy)

        if dist <= self.radius or dist == 0.0:
            return 0.0, 0.0, dist

        # Continuous smooth ramp: scales remaining distance proportionally
        # without introducing a step discontinuity at the deadzone boundary
        scaled_dist = dist - self.radius
        ratio = scaled_dist / dist

        return dx * ratio, dy * ratio, dist
