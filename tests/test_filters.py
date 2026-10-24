"""Unit tests for smoothing and deadzone filters."""
import math
import pytest
from src.filters import LowPassFilter, OneEuroFilter, PointOneEuroFilter, DeadZoneFilter


def test_low_pass_filter_initialization():
    lpf = LowPassFilter(alpha=0.5)
    assert lpf.last_value() is None
    val = lpf.filter(10.0)
    assert val == 10.0
    val2 = lpf.filter(20.0)
    assert val2 == 15.0  # 0.5 * 20 + 0.5 * 10
    lpf.reset()
    assert lpf.last_value() is None


def test_one_euro_filter_smoothing_at_rest():
    """At low speed/rest, the filter should strongly attenuate jitter."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01, d_cutoff=1.0)
    base = 100.0
    t = 0.0

    # Feed slight noisy perturbations around 100.0
    results = []
    for i in range(20):
        t += 0.033
        noise = (-1) ** i * 0.5
        filtered = f.filter(base + noise, timestamp=t)
        results.append(filtered)

    # Filtered output variance should be significantly smaller than raw noise
    raw_deviations = [0.5 for _ in range(20)]
    filtered_deviations = [abs(r - base) for r in results[5:]]
    assert max(filtered_deviations) < max(raw_deviations)


def test_one_euro_filter_responsiveness_on_fast_motion():
    """On fast sudden motion, beta parameter increases cutoff to avoid lag."""
    f = OneEuroFilter(min_cutoff=0.1, beta=1.0, d_cutoff=1.0)
    t = 0.0
    # Settle at 0
    for _ in range(10):
        t += 0.033
        f.filter(0.0, timestamp=t)

    # Sudden jump to 100
    t += 0.033
    jump_filtered = f.filter(100.0, timestamp=t)
    # High beta should allow quick jump response
    assert jump_filtered > 20.0


def test_point_one_euro_filter():
    pf = PointOneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    fx, fy = pf.filter(10.0, 20.0, timestamp=1.0)
    assert fx == 10.0 and fy == 20.0

    fx2, fy2 = pf.filter(12.0, 22.0, timestamp=1.05)
    assert 10.0 < fx2 < 12.0
    assert 20.0 < fy2 < 22.0

    pf.update_params(min_cutoff=2.0, beta=0.1, d_cutoff=1.5)
    assert pf._filter_x.min_cutoff == 2.0
    assert pf._filter_y.min_cutoff == 2.0


def test_deadzone_filter_inside_radius():
    dz = DeadZoneFilter(radius=0.05)
    neutral_x, neutral_y = 0.5, 0.5

    # Center exactly
    dx, dy, dist = dz.apply(0.5, 0.5, neutral_x, neutral_y)
    assert dx == 0.0 and dy == 0.0 and dist == 0.0

    # Small displacement inside deadzone (0.02 < 0.05)
    dx, dy, dist = dz.apply(0.51, 0.51, neutral_x, neutral_y)
    assert dx == 0.0 and dy == 0.0
    assert dist == pytest.approx(math.hypot(0.01, 0.01))


def test_deadzone_filter_outside_radius():
    dz = DeadZoneFilter(radius=0.05)
    neutral_x, neutral_y = 0.5, 0.5

    # Move outside (displacement 0.10 > 0.05)
    dx, dy, dist = dz.apply(0.60, 0.50, neutral_x, neutral_y)
    assert dist == pytest.approx(0.10)
    assert dx > 0.0
    assert dy == 0.0
    assert dx < 0.10  # Scaled smoothly
