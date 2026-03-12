from typing import Optional

import pandas as pd
import yfinance as yf

from .utils import fmt_big_num, fmt_num


def get_fundamentals(
    ticker: str,
    info: Optional[dict] = None,
    fast_info: Optional[dict] = None,
) -> dict:
    if info is None or fast_info is None:
        stock = yf.Ticker(ticker)
        if fast_info is None:
            try:
                fast_info = dict(stock.fast_info)
            except Exception:
                fast_info = {}
        if info is None:
            try:
                info = stock.info
            except Exception:
                info = {}
    else:
        fast_info = dict(fast_info)

    dividend_yield = info.get("dividendYield")
    if dividend_yield is not None and pd.notna(dividend_yield):
        dividend_yield = float(dividend_yield) * 100

    market_cap = fast_info.get("marketCap", info.get("marketCap"))
    avg_volume = fast_info.get("tenDayAverageVolume", info.get("averageVolume"))

    return {
        "종목명": info.get("shortName") or info.get("longName") or ticker,
        "시가총액": fmt_big_num(market_cap),
        "PER": fmt_num(info.get("trailingPE")),
        "PBR": fmt_num(info.get("priceToBook")),
        "52주 최고": fmt_num(info.get("fiftyTwoWeekHigh")),
        "52주 최저": fmt_num(info.get("fiftyTwoWeekLow")),
        "평균거래량": fmt_big_num(avg_volume),
        "배당수익률(%)": fmt_num(dividend_yield),
        "섹터": info.get("sector", "-"),
        "산업": info.get("industry", "-"),
    }


def _normalize_financial_df(fin_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if fin_df is None or fin_df.empty:
        return pd.DataFrame()
    df = fin_df.copy()
    if isinstance(df.columns, pd.DatetimeIndex):
        df = df.sort_index(axis=1)
    return df


def _find_row(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    if df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _yoy_from_series(s: Optional[pd.Series]) -> Optional[float]:
    if s is None or len(s) < 2:
        return None
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) < 2:
        return None
    prev = vals.iloc[-2]
    curr = vals.iloc[-1]
    if prev == 0:
        return None
    return (curr - prev) / abs(prev) * 100.0


def get_financial_metrics(ticker: str) -> dict:
    stock = yf.Ticker(ticker)

    income_stmt = pd.DataFrame()
    cashflow = pd.DataFrame()
    balance_sheet = pd.DataFrame()
    quarterly_income_stmt = pd.DataFrame()

    try:
        income_stmt = _normalize_financial_df(stock.income_stmt)
    except Exception:
        pass

    try:
        quarterly_income_stmt = _normalize_financial_df(stock.quarterly_income_stmt)
    except Exception:
        pass

    try:
        cashflow = _normalize_financial_df(stock.cashflow)
    except Exception:
        pass

    try:
        balance_sheet = _normalize_financial_df(stock.balance_sheet)
    except Exception:
        pass

    revenue_row = _find_row(income_stmt, ["Total Revenue", "Operating Revenue", "Revenue"])
    q_revenue_row = _find_row(quarterly_income_stmt, ["Total Revenue", "Operating Revenue", "Revenue"])
    net_income_row = _find_row(income_stmt, ["Net Income", "Net Income Common Stockholders"])
    q_net_income_row = _find_row(quarterly_income_stmt, ["Net Income", "Net Income Common Stockholders"])
    op_cf_row = _find_row(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    fcf_row = _find_row(cashflow, ["Free Cash Flow"])
    capex_row = _find_row(cashflow, ["Capital Expenditure", "Capital Expenditures"])
    total_liab_row = _find_row(balance_sheet, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
    equity_row = _find_row(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"])
    gross_profit_row = _find_row(income_stmt, ["Gross Profit"])
    operating_income_row = _find_row(income_stmt, ["Operating Income"])
    shares_row = _find_row(balance_sheet, ["Ordinary Shares Number", "Share Issued"])
    current_assets_row = _find_row(balance_sheet, ["Current Assets"])
    current_liab_row = _find_row(balance_sheet, ["Current Liabilities"])

    revenue_yoy = _yoy_from_series(revenue_row)
    q_revenue_yoy = _yoy_from_series(q_revenue_row)
    net_income_yoy = _yoy_from_series(net_income_row)
    q_net_income_yoy = _yoy_from_series(q_net_income_row)

    op_cf = None
    fcf = None
    debt_ratio = None
    gross_margin = None
    operating_margin = None
    current_ratio = None
    share_change_yoy = None
    earnings_quality = None
    roe = None

    if op_cf_row is not None:
        vals = pd.to_numeric(op_cf_row, errors="coerce").dropna()
        if len(vals) > 0:
            op_cf = float(vals.iloc[-1])

    if fcf_row is not None:
        vals = pd.to_numeric(fcf_row, errors="coerce").dropna()
        if len(vals) > 0:
            fcf = float(vals.iloc[-1])
    elif op_cf is not None and capex_row is not None:
        cap_vals = pd.to_numeric(capex_row, errors="coerce").dropna()
        if len(cap_vals) > 0:
            capex = float(cap_vals.iloc[-1])
            fcf = op_cf + capex

    if total_liab_row is not None and equity_row is not None:
        liab_vals = pd.to_numeric(total_liab_row, errors="coerce").dropna()
        eq_vals = pd.to_numeric(equity_row, errors="coerce").dropna()
        if len(liab_vals) > 0 and len(eq_vals) > 0 and eq_vals.iloc[-1] != 0:
            debt_ratio = float(liab_vals.iloc[-1] / eq_vals.iloc[-1] * 100)

    if revenue_row is not None and gross_profit_row is not None:
        rev_vals = pd.to_numeric(revenue_row, errors="coerce").dropna()
        gp_vals = pd.to_numeric(gross_profit_row, errors="coerce").dropna()
        if len(rev_vals) > 0 and len(gp_vals) > 0 and rev_vals.iloc[-1] != 0:
            gross_margin = float(gp_vals.iloc[-1] / rev_vals.iloc[-1] * 100)

    if revenue_row is not None and operating_income_row is not None:
        rev_vals = pd.to_numeric(revenue_row, errors="coerce").dropna()
        opi_vals = pd.to_numeric(operating_income_row, errors="coerce").dropna()
        if len(rev_vals) > 0 and len(opi_vals) > 0 and rev_vals.iloc[-1] != 0:
            operating_margin = float(opi_vals.iloc[-1] / rev_vals.iloc[-1] * 100)

    if current_assets_row is not None and current_liab_row is not None:
        ca_vals = pd.to_numeric(current_assets_row, errors="coerce").dropna()
        cl_vals = pd.to_numeric(current_liab_row, errors="coerce").dropna()
        if len(ca_vals) > 0 and len(cl_vals) > 0 and cl_vals.iloc[-1] != 0:
            current_ratio = float(ca_vals.iloc[-1] / cl_vals.iloc[-1])

    if shares_row is not None:
        sh_vals = pd.to_numeric(shares_row, errors="coerce").dropna()
        if len(sh_vals) >= 2 and sh_vals.iloc[-2] != 0:
            share_change_yoy = float((sh_vals.iloc[-1] - sh_vals.iloc[-2]) / sh_vals.iloc[-2] * 100)

    if op_cf is not None and net_income_row is not None:
        ni_vals = pd.to_numeric(net_income_row, errors="coerce").dropna()
        if len(ni_vals) > 0 and ni_vals.iloc[-1] != 0:
            earnings_quality = float(op_cf / ni_vals.iloc[-1])

    if equity_row is not None and net_income_row is not None:
        eq_vals = pd.to_numeric(equity_row, errors="coerce").dropna()
        ni_vals = pd.to_numeric(net_income_row, errors="coerce").dropna()
        if len(eq_vals) > 0 and len(ni_vals) > 0 and eq_vals.iloc[-1] != 0:
            roe = float(ni_vals.iloc[-1] / eq_vals.iloc[-1] * 100)

    return {
        "annual_revenue_yoy_pct": revenue_yoy,
        "quarterly_revenue_yoy_pct": q_revenue_yoy,
        "annual_net_income_yoy_pct": net_income_yoy,
        "quarterly_net_income_yoy_pct": q_net_income_yoy,
        "operating_cash_flow": op_cf,
        "free_cash_flow": fcf,
        "debt_to_equity_pct": debt_ratio,
        "gross_margin_pct": gross_margin,
        "operating_margin_pct": operating_margin,
        "current_ratio": current_ratio,
        "share_change_yoy_pct": share_change_yoy,
        "earnings_quality_ratio": earnings_quality,
        "roe_pct": roe,
    }
