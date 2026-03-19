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


def _latest_val(row: Optional[pd.Series]) -> Optional[float]:
    if row is None:
        return None
    vals = pd.to_numeric(row, errors="coerce").dropna()
    if len(vals) == 0:
        return None
    return float(vals.iloc[-1])


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

    # --- 행 추출 ---
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
    operating_income_row = _find_row(income_stmt, ["Operating Income", "EBIT"])
    shares_row = _find_row(balance_sheet, ["Ordinary Shares Number", "Share Issued"])
    current_assets_row = _find_row(balance_sheet, ["Current Assets"])
    current_liab_row = _find_row(balance_sheet, ["Current Liabilities"])
    interest_expense_row = _find_row(income_stmt, ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"])
    inventory_row = _find_row(balance_sheet, ["Inventory"])
    cash_row = _find_row(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
    total_assets_row = _find_row(balance_sheet, ["Total Assets"])
    ebitda_row = _find_row(income_stmt, ["EBITDA", "Normalized EBITDA"])

    # --- YoY 성장률 ---
    revenue_yoy = _yoy_from_series(revenue_row)
    q_revenue_yoy = _yoy_from_series(q_revenue_row)
    net_income_yoy = _yoy_from_series(net_income_row)
    q_net_income_yoy = _yoy_from_series(q_net_income_row)
    ocf_yoy = _yoy_from_series(op_cf_row)

    # --- 기본 지표 ---
    op_cf = _latest_val(op_cf_row)
    revenue_latest = _latest_val(revenue_row)
    net_income_latest = _latest_val(net_income_row)
    operating_income_latest = _latest_val(operating_income_row)
    equity_latest = _latest_val(equity_row)
    total_liab_latest = _latest_val(total_liab_row)
    total_assets_latest = _latest_val(total_assets_row)
    current_assets_latest = _latest_val(current_assets_row)
    current_liab_latest = _latest_val(current_liab_row)
    cash_latest = _latest_val(cash_row)
    ebitda_latest = _latest_val(ebitda_row)

    fcf = _latest_val(fcf_row)
    if fcf is None and op_cf is not None:
        capex_val = _latest_val(capex_row)
        if capex_val is not None:
            fcf = op_cf + capex_val

    capex_latest = _latest_val(capex_row)

    # --- 파생 지표 계산 ---
    debt_ratio = None
    if total_liab_latest is not None and equity_latest is not None and equity_latest != 0:
        debt_ratio = float(total_liab_latest / equity_latest * 100)

    gross_margin = None
    gp_latest = _latest_val(gross_profit_row)
    if revenue_latest is not None and gp_latest is not None and revenue_latest != 0:
        gross_margin = float(gp_latest / revenue_latest * 100)

    operating_margin = None
    if revenue_latest is not None and operating_income_latest is not None and revenue_latest != 0:
        operating_margin = float(operating_income_latest / revenue_latest * 100)

    net_margin = None
    if revenue_latest is not None and net_income_latest is not None and revenue_latest != 0:
        net_margin = float(net_income_latest / revenue_latest * 100)

    current_ratio = None
    if current_assets_latest is not None and current_liab_latest is not None and current_liab_latest != 0:
        current_ratio = float(current_assets_latest / current_liab_latest)

    quick_ratio = None
    inventory_latest = _latest_val(inventory_row)
    if current_assets_latest is not None and current_liab_latest is not None and current_liab_latest != 0:
        inv = inventory_latest if inventory_latest is not None else 0.0
        quick_ratio = float((current_assets_latest - inv) / current_liab_latest)

    share_change_yoy = None
    if shares_row is not None:
        sh_vals = pd.to_numeric(shares_row, errors="coerce").dropna()
        if len(sh_vals) >= 2 and sh_vals.iloc[-2] != 0:
            share_change_yoy = float((sh_vals.iloc[-1] - sh_vals.iloc[-2]) / sh_vals.iloc[-2] * 100)

    earnings_quality = None
    if op_cf is not None and net_income_latest is not None and net_income_latest != 0:
        earnings_quality = float(op_cf / net_income_latest)

    roe = None
    if equity_latest is not None and net_income_latest is not None and equity_latest != 0:
        roe = float(net_income_latest / equity_latest * 100)

    roa = None
    if total_assets_latest is not None and net_income_latest is not None and total_assets_latest != 0:
        roa = float(net_income_latest / total_assets_latest * 100)

    interest_coverage = None
    interest_latest = _latest_val(interest_expense_row)
    if operating_income_latest is not None and interest_latest is not None and interest_latest != 0:
        interest_coverage = float(abs(operating_income_latest / interest_latest))

    net_debt_to_ebitda = None
    if ebitda_latest is not None and ebitda_latest != 0 and total_liab_latest is not None:
        net_debt = total_liab_latest - (cash_latest if cash_latest is not None else 0.0)
        net_debt_to_ebitda = float(net_debt / ebitda_latest)

    fcf_margin = None
    if fcf is not None and revenue_latest is not None and revenue_latest != 0:
        fcf_margin = float(fcf / revenue_latest * 100)

    capex_to_revenue = None
    if capex_latest is not None and revenue_latest is not None and revenue_latest != 0:
        capex_to_revenue = float(abs(capex_latest) / revenue_latest * 100)

    # --- 밸류에이션 (info 기반) ---
    try:
        info = stock.info
    except Exception:
        info = {}

    ev_to_ebitda = None
    peg_ratio = None
    price_to_sales = None
    price_to_fcf = None
    fcf_yield = None
    payout_ratio = None
    market_cap = None

    ev_ebitda_raw = info.get("enterpriseToEbitda")
    if ev_ebitda_raw is not None and not pd.isna(ev_ebitda_raw):
        ev_to_ebitda = float(ev_ebitda_raw)

    peg_raw = info.get("pegRatio")
    if peg_raw is not None and not pd.isna(peg_raw):
        peg_ratio = float(peg_raw)

    ps_raw = info.get("priceToSalesTrailing12Months")
    if ps_raw is not None and not pd.isna(ps_raw):
        price_to_sales = float(ps_raw)

    payout_raw = info.get("payoutRatio")
    if payout_raw is not None and not pd.isna(payout_raw):
        payout_ratio = float(payout_raw * 100)

    mc_raw = info.get("marketCap")
    if mc_raw is not None and not pd.isna(mc_raw):
        market_cap = float(mc_raw)

    if fcf is not None and market_cap is not None and market_cap != 0:
        price_to_fcf = float(market_cap / fcf) if fcf != 0 else None
        fcf_yield = float(fcf / market_cap * 100)

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
        "net_margin_pct": net_margin,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "share_change_yoy_pct": share_change_yoy,
        "earnings_quality_ratio": earnings_quality,
        "roe_pct": roe,
        "roa_pct": roa,
        "interest_coverage": interest_coverage,
        "net_debt_to_ebitda": net_debt_to_ebitda,
        "fcf_margin_pct": fcf_margin,
        "capex_to_revenue_pct": capex_to_revenue,
        "ocf_yoy_pct": ocf_yoy,
        "ev_to_ebitda": ev_to_ebitda,
        "peg_ratio": peg_ratio,
        "price_to_sales": price_to_sales,
        "price_to_fcf": price_to_fcf,
        "fcf_yield_pct": fcf_yield,
        "payout_ratio_pct": payout_ratio,
    }
