from typing import Optional

import pandas as pd
import yfinance as yf


SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Utilities": "XLU",
}


def get_sector_comparison(ticker: str, info: Optional[dict] = None) -> dict:
    result: dict = {
        "sector": "-",
        "industry": "-",
        "sector_etf": None,
        "ticker_return_3m": None,
        "sector_return_3m": None,
        "return_vs_sector_3m": None,
        "ticker_return_6m": None,
        "sector_return_6m": None,
        "return_vs_sector_6m": None,
        "ticker_return_1y": None,
        "sector_return_1y": None,
        "return_vs_sector_1y": None,
        "return_vs_sector": None,
        "sector_per": None,
        "market_weight_pct": None,
        "target_per": None,
        "target_forward_pe": None,
        "target_op_margin": None,
        "target_roe": None,
    }

    if info is None:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            return result

    result["sector"] = info.get("sector") or "-"
    result["industry"] = info.get("industry") or "-"

    per = info.get("trailingPE")
    if per and not pd.isna(per):
        result["target_per"] = round(float(per), 1)

    fpe = info.get("forwardPE")
    if fpe and not pd.isna(fpe):
        result["target_forward_pe"] = round(float(fpe), 1)

    op_margin = info.get("operatingMargins")
    if op_margin is not None and not pd.isna(op_margin):
        result["target_op_margin"] = round(float(op_margin) * 100, 1)

    roe = info.get("returnOnEquity")
    if roe is not None and not pd.isna(roe):
        result["target_roe"] = round(float(roe) * 100, 1)

    sector_etf = SECTOR_ETF_MAP.get(result["sector"])
    if sector_etf:
        result["sector_etf"] = sector_etf
        _compare_returns(ticker, sector_etf, result)
        _get_sector_etf_per(sector_etf, result)

    industry_key = info.get("industryKey")
    if industry_key:
        _get_market_weight(ticker, industry_key, result)

    return result


def _set_relative_returns(result: dict, key: str, ticker_series: pd.Series, etf_series: pd.Series, bars: int):
    if len(ticker_series) <= bars or len(etf_series) <= bars:
        return

    ticker_return = (float(ticker_series.iloc[-1]) / float(ticker_series.iloc[-bars - 1]) - 1) * 100
    sector_return = (float(etf_series.iloc[-1]) / float(etf_series.iloc[-bars - 1]) - 1) * 100
    result[f"ticker_return_{key}"] = round(ticker_return, 1)
    result[f"sector_return_{key}"] = round(sector_return, 1)
    result[f"return_vs_sector_{key}"] = round(ticker_return - sector_return, 1)



def _compare_returns(ticker: str, etf: str, result: dict):
    try:
        data = yf.download([ticker, etf], period="1y", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data

        cols_upper = {c.upper(): c for c in closes.columns}
        t_col = cols_upper.get(ticker.upper())
        e_col = cols_upper.get(etf.upper())
        if t_col is None or e_col is None:
            return

        ticker_series = closes[t_col].dropna()
        etf_series = closes[e_col].dropna()
        if len(ticker_series) < 2 or len(etf_series) < 2:
            return

        _set_relative_returns(result, "3m", ticker_series, etf_series, 63)
        _set_relative_returns(result, "6m", ticker_series, etf_series, 126)

        result["ticker_return_1y"] = round((float(ticker_series.iloc[-1]) / float(ticker_series.iloc[0]) - 1) * 100, 1)
        result["sector_return_1y"] = round((float(etf_series.iloc[-1]) / float(etf_series.iloc[0]) - 1) * 100, 1)
        result["return_vs_sector_1y"] = round(result["ticker_return_1y"] - result["sector_return_1y"], 1)
        result["return_vs_sector"] = result["return_vs_sector_1y"]
    except Exception:
        pass


def _get_sector_etf_per(etf: str, result: dict):
    try:
        etf_info = yf.Ticker(etf).info
        per = etf_info.get("trailingPE")
        if per and not pd.isna(per) and per > 0:
            result["sector_per"] = round(float(per), 1)
    except Exception:
        pass



def _get_market_weight(ticker: str, industry_key: str, result: dict):
    try:
        industry = yf.Industry(industry_key)
        top_cos = industry.top_companies
        if top_cos is None or top_cos.empty:
            return

        sym_col = next((c for c in ["symbol", "ticker"] if c in top_cos.columns), None)
        if sym_col is None:
            return

        match = top_cos[top_cos[sym_col].str.upper() == ticker.upper()]
        if match.empty:
            return

        for w_col in ["market_weight", "weight"]:
            if w_col in top_cos.columns:
                w = match.iloc[0].get(w_col)
                if w is not None and not pd.isna(w):
                    result["market_weight_pct"] = round(float(w) * 100, 2)
                break
    except Exception:
        pass
