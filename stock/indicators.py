import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for w in (5, 20, 60, 120, 200):
        df[f"MA_{w}"] = df["Close"].rolling(window=w, min_periods=w).mean()

    # RSI (Wilder)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    df["BB_mid"] = df["Close"].rolling(20, min_periods=20).mean()
    bb_std = df["Close"].rolling(20, min_periods=20).std(ddof=0)
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    df["BB_width_pct"] = ((df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]) * 100

    # 52주 고점/저점
    df["HIGH_52W"] = df["High"].rolling(252, min_periods=20).max()
    df["LOW_52W"] = df["Low"].rolling(252, min_periods=20).min()

    # ATR(14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14, min_periods=14).mean()
    df["ATR_PCT"] = df["ATR_14"] / df["Close"] * 100

    # ADX(14)
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr14 = tr.rolling(14, min_periods=14).sum()
    plus_di = 100 * plus_dm.rolling(14, min_periods=14).sum() / tr14.replace(0, 1e-10)
    minus_di = 100 * minus_dm.rolling(14, min_periods=14).sum() / tr14.replace(0, 1e-10)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)) * 100
    df["ADX_14"] = dx.rolling(14, min_periods=14).mean()

    # OBV
    direction = df["Close"].diff().fillna(0)
    obv_step = direction.map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["OBV"] = (obv_step * df["Volume"]).cumsum()
    df["OBV_MA_20"] = df["OBV"].rolling(20, min_periods=20).mean()

    return df
