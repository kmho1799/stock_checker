from typing import Optional

import pandas as pd


def detect_pivots(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> tuple[list[tuple[pd.Timestamp, float]], list[tuple[pd.Timestamp, float]]]:
    highs = []
    lows = []

    if len(df) < left + right + 1:
        return highs, lows

    high_arr = df["High"].values
    low_arr = df["Low"].values
    idx = df.index

    for i in range(left, len(df) - right):
        win_high = high_arr[i - left:i + right + 1]
        win_low = low_arr[i - left:i + right + 1]

        center_high = high_arr[i]
        center_low = low_arr[i]

        if center_high == win_high.max():
            if (win_high == center_high).sum() == 1:
                highs.append((idx[i], float(center_high)))

        if center_low == win_low.min():
            if (win_low == center_low).sum() == 1:
                lows.append((idx[i], float(center_low)))

    return highs, lows


def cluster_levels(levels: list[tuple[pd.Timestamp, float]], tolerance_pct: float = 1.5) -> list[float]:
    if not levels:
        return []

    prices = sorted([p for _, p in levels])
    clusters = [[prices[0]]]

    for p in prices[1:]:
        prev_mean = sum(clusters[-1]) / len(clusters[-1])
        if abs(p - prev_mean) / prev_mean * 100 <= tolerance_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    return [sum(c) / len(c) for c in clusters]


def get_support_resistance_pivot(
    df: pd.DataFrame,
    lookback_days: int,
    current_price: float,
    left: int = 3,
    right: int = 3,
    tolerance_pct: float = 1.5,
) -> tuple[Optional[float], Optional[float], list[float], list[float]]:
    if len(df) < max(lookback_days, left + right + 5):
        return None, None, [], []

    window = df.tail(lookback_days).copy()
    highs, lows = detect_pivots(window, left=left, right=right)

    resistance_levels = cluster_levels(highs, tolerance_pct=tolerance_pct)
    support_levels = cluster_levels(lows, tolerance_pct=tolerance_pct)

    below = [x for x in support_levels if x < current_price]
    above = [x for x in resistance_levels if x > current_price]

    nearest_support = max(below) if below else None
    nearest_resistance = min(above) if above else None

    return nearest_support, nearest_resistance, support_levels, resistance_levels
