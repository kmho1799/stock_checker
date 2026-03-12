import pandas as pd


def fmt_big_num(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        value = float(value)
    except Exception:
        return str(value)

    if value >= 1e12:
        return f"{value/1e12:.2f}T"
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    if value >= 1e6:
        return f"{value/1e6:.2f}M"
    if value >= 1e3:
        return f"{value/1e3:.2f}K"
    return f"{value:.0f}"


def fmt_num(value, digits: int = 2):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None
