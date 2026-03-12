import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"'{ticker}' 종목 데이터를 가져올 수 없습니다.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df.dropna(subset=["Close"]).copy()
    return df


def get_extended_market_price(ticker: str) -> dict:
    """
    프리마켓/애프터마켓 포함 최신 분봉 데이터 조회
    반환 예:
    {
        "price": 123.45,
        "time": Timestamp(...),
        "session": "regular" | "premarket" | "aftermarket" | "unknown",
        "regular_close": 122.10
    }
    """
    stock = yf.Ticker(ticker)

    try:
        intraday = stock.history(
            period="5d",
            interval="1m",
            prepost=True,
            auto_adjust=False,
            actions=False,
            repair=False,
        )
    except Exception:
        intraday = pd.DataFrame()

    result = {
        "price": None,
        "time": None,
        "session": "regular",
        "regular_close": None,
    }

    if intraday.empty:
        return result

    intraday = intraday.dropna(subset=["Close"]).copy()
    if intraday.empty:
        return result

    latest = intraday.iloc[-1]
    latest_ts = intraday.index[-1]
    latest_price = float(latest["Close"])

    try:
        same_day = intraday[intraday.index.date == latest_ts.date()].copy()
    except Exception:
        same_day = intraday.copy()

    regular_close = None
    if not same_day.empty:
        try:
            hhmm = same_day.index.hour * 100 + same_day.index.minute
            regular_session = same_day[(hhmm >= 930) & (hhmm <= 1600)]
            if not regular_session.empty:
                regular_close = float(regular_session.iloc[-1]["Close"])
        except Exception:
            pass

    session = "regular"
    try:
        h = latest_ts.hour
        m = latest_ts.minute
        hhmm = h * 100 + m

        if 400 <= hhmm < 930:
            session = "premarket"
        elif 930 <= hhmm <= 1600:
            session = "regular"
        elif 1600 < hhmm <= 2000:
            session = "aftermarket"
        else:
            session = "unknown"
    except Exception:
        session = "unknown"

    result["price"] = latest_price
    result["time"] = latest_ts
    result["session"] = session
    result["regular_close"] = regular_close
    return result
