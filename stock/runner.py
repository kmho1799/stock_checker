from typing import Any, Literal, Optional

import pandas as pd
import yfinance as yf

from .analysis import (
    analyze_long_term_fundamentals,
    analyze_sector_comparison,
    analyze_speed,
    analyze_support_resistance,
    analyze_trend,
    analyze_volatility,
    analyze_volume,
)
from .data import fetch_stock_data, get_extended_market_price
from .fundamentals import get_financial_metrics, get_fundamentals
from .indicators import add_technical_indicators
from .scoring import combine_scores, score_fundamentals, score_technical
from .sector import get_sector_comparison


Term = Literal["short", "medium", "long", "all"]


EMPTY_SECTOR_COMP = {
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


def _safe_ticker_info(ticker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        stock = yf.Ticker(ticker)
        return dict(stock.fast_info), stock.info
    except Exception:
        return {}, {}



def _safe_fetch_fundamentals(ticker: str, info: dict[str, Any], fast_info: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    try:
        return get_fundamentals(ticker, info=info, fast_info=fast_info), None
    except Exception as exc:
        return {}, str(exc)



def _safe_fetch_financial_metrics(ticker: str) -> tuple[dict[str, Any], Optional[str]]:
    try:
        return get_financial_metrics(ticker), None
    except Exception as exc:
        return {}, str(exc)



def _safe_fetch_extended_market_price(ticker: str) -> tuple[dict[str, Any], Optional[str]]:
    try:
        return get_extended_market_price(ticker), None
    except Exception as exc:
        return {"price": None, "time": None, "session": "regular", "regular_close": None}, str(exc)



def _safe_fetch_sector_comparison(ticker: str, info: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    try:
        return get_sector_comparison(ticker, info=info), None
    except Exception as exc:
        return dict(EMPTY_SECTOR_COMP), str(exc)



def _normalize_timestamp(value: Any) -> Any:
    try:
        return value.tz_localize(None)
    except Exception:
        return value



def _normalize_term(term: Term) -> str:
    return "medium" if term == "all" else term



def _build_price_summary(df: pd.DataFrame, ext: dict[str, Any]) -> dict[str, Any]:
    latest = df.iloc[-1]
    daily_close = float(latest["Close"])

    current_price = daily_close
    market_state = "REGULAR_CLOSE"
    compare_close = daily_close
    ext_time = ext.get("time")

    if ext.get("price") is not None:
        ext_price = float(ext["price"])
        session = ext.get("session", "regular")
        if ext.get("regular_close") is not None:
            compare_close = float(ext["regular_close"])

        if session in ("premarket", "aftermarket", "unknown"):
            current_price = ext_price
            if session == "premarket":
                market_state = "PREMARKET"
            elif session == "aftermarket":
                market_state = "AFTERMARKET"
            else:
                market_state = "EXTENDED"

    diff_pct = None
    if current_price != compare_close:
        diff_pct = (current_price - compare_close) / compare_close * 100

    moving_averages = {}
    for window in (5, 20, 60, 120, 200):
        value = latest.get(f"MA_{window}")
        if value is not None and not pd.isna(value):
            moving_averages[window] = float(value)

    rsi_value = latest.get("RSI")
    return {
        "latest_date": _normalize_timestamp(latest.name),
        "daily_close": daily_close,
        "current_price": current_price,
        "market_state": market_state,
        "compare_close": compare_close,
        "ext_time": _normalize_timestamp(ext_time) if ext_time is not None else None,
        "diff_pct": diff_pct,
        "moving_averages": moving_averages,
        "rsi": None if pd.isna(rsi_value) else rsi_value,
    }



def _build_term_sections(df: pd.DataFrame, fin: dict[str, Any], current_price: float, term: Term) -> list[dict[str, Any]]:
    section_names = {
        "trend": "추세 (이동평균선)",
        "volume": "힘 (거래량)",
        "speed": "속도 (RSI, MACD)",
        "volatility": "변동성 (볼린저밴드)",
        "sr": "가격대 (pivot 지지·저항)",
        "fundamentals": "장기 펀더멘털",
    }
    term_labels = {"short": "단기", "medium": "중기", "long": "장기"}
    terms_to_show = ["short", "medium", "long"] if term == "all" else [term]

    sections = []
    for current_term in terms_to_show:
        items = [
            {"name": section_names["trend"], "lines": analyze_trend(df, current_term, current_price=current_price)},
            {"name": section_names["volume"], "lines": analyze_volume(df, current_term)},
            {"name": section_names["speed"], "lines": analyze_speed(df, current_term)},
            {"name": section_names["volatility"], "lines": analyze_volatility(df, current_term, current_price=current_price)},
            {"name": section_names["sr"], "lines": analyze_support_resistance(df, current_term, current_price=current_price)},
        ]
        if current_term == "long":
            items.append({
                "name": section_names["fundamentals"],
                "lines": analyze_long_term_fundamentals(df, fin, current_price=current_price),
            })
        sections.append({"term": current_term, "label": term_labels[current_term], "items": items})

    return sections



def _build_recent_ohlc(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if len(df) < 5:
        return None

    subset = df[["Open", "High", "Low", "Close", "Volume"]].tail(5).copy()
    subset["Volume"] = subset["Volume"].apply(
        lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else (f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}")
    )
    return subset



def build_analysis_report(ticker: str, period: str = "2y", show_chart_data: bool = True, term: Term = "all") -> dict[str, Any]:
    ticker = ticker.upper()
    score_term = _normalize_term(term)
    fast_info, info = _safe_ticker_info(ticker)

    base_info, base_info_error = _safe_fetch_fundamentals(ticker, info=info, fast_info=fast_info)

    df = fetch_stock_data(ticker, period=period)
    df = add_technical_indicators(df)

    fin, fin_error = _safe_fetch_financial_metrics(ticker)
    ext, ext_error = _safe_fetch_extended_market_price(ticker)
    sector_comp, sector_error = _safe_fetch_sector_comparison(ticker, info=info)

    price_summary = _build_price_summary(df, ext)
    sections = _build_term_sections(df, fin, current_price=price_summary["current_price"], term=term)

    tech_score, tech_reasons, tech_factors = score_technical(
        df,
        current_price=price_summary["current_price"],
        term=score_term,
        sector_comp=sector_comp,
    )
    fund_score, fund_reasons, fund_factors = score_fundamentals(
        fin,
        df,
        current_price=price_summary["current_price"],
        sector_comp=sector_comp,
    )
    total_score, total_label = combine_scores(tech_score, fund_score, term=score_term)

    warnings = []
    if base_info_error:
        warnings.append(f"기본 정보 조회 실패: {base_info_error}")
    if fin_error:
        warnings.append(f"재무 지표 조회 실패: {fin_error}")
    if ext_error:
        warnings.append(f"연장장 데이터 조회 실패: {ext_error}")
    if sector_error:
        warnings.append(f"섹터 비교 조회 실패: {sector_error}")

    return {
        "ticker": ticker,
        "period": period,
        "show_chart_data": show_chart_data,
        "warnings": warnings,
        "base_info": base_info,
        "sector_lines": analyze_sector_comparison(sector_comp),
        "price_summary": price_summary,
        "sections": sections,
        "scores": {
            "technical": tech_score,
            "fundamentals": fund_score,
            "total": total_score,
            "label": total_label,
            "term": score_term,
            "technical_reasons": tech_reasons,
            "fundamental_reasons": fund_reasons,
            "technical_factors": tech_factors,
            "fundamental_factors": fund_factors,
        },
        "recent_ohlc": _build_recent_ohlc(df) if show_chart_data else None,
    }



def _print_header(report: dict[str, Any]):
    print(f"\n{'='*60}")
    print(f"  {report['ticker']} 투자 분석 (Yahoo Finance)")
    print(f"{'='*60}\n")



def _print_warnings(report: dict[str, Any]):
    for warning in report["warnings"]:
        print(f"[참고] {warning}")
    if report["warnings"]:
        print()



def _print_base_info(report: dict[str, Any]):
    print("[기본 정보]")
    for key, value in report["base_info"].items():
        if value is not None and value != "-":
            print(f"  {key}: {value}")
    print()



def _print_sector_info(report: dict[str, Any]):
    print("[섹터/산업 대비 분석]")
    for line in report["sector_lines"]:
        print(f"  {line}")
    print()



def _print_price_summary(report: dict[str, Any]):
    summary = report["price_summary"]

    print("[최근 가격]")
    latest_date = summary["latest_date"]
    print(f"  일봉 기준 일자: {latest_date.date() if hasattr(latest_date, 'date') else latest_date}")
    print(f"  정규장 종가: {summary['daily_close']:.2f}")
    print(f"  분석 기준 가격: {summary['current_price']:.2f} ({summary['market_state']})")

    if summary["ext_time"] is not None:
        print(f"  연장장 체결 시각: {summary['ext_time']}")

    if summary["diff_pct"] is not None:
        print(f"  정규장 종가 대비: {summary['diff_pct']:+.2f}%")

    for window, value in summary["moving_averages"].items():
        print(f"  MA({window}): {value:.2f}")

    if summary["rsi"] is not None:
        print(f"  RSI: {summary['rsi']:.1f}")
    print()



def _print_sections(report: dict[str, Any]):
    for section in report["sections"]:
        print(f"[{section['label']} 분석]")
        for item in section["items"]:
            print(f"  ▶ {item['name']}")
            for line in item["lines"]:
                print(f"    - {line}")
        print()



def _print_scores(report: dict[str, Any]):
    scores = report["scores"]

    print("[매수/매도 신호 점수]")
    print(f"  점수 기준 기간: {scores['term']}")
    print(f"  기술적 점수: {scores['technical']}/100")
    print(f"  펀더멘털 점수: {scores['fundamentals']}/100")
    print(f"  종합 점수: {scores['total']}/100 → {scores['label']}")
    print("  기술 팩터:")
    for name, value in scores["technical_factors"].items():
        print(f"    - {name}: {value}")
    print("  재무 팩터:")
    for name, value in scores["fundamental_factors"].items():
        print(f"    - {name}: {value}")
    print("  기술적 근거:")
    for reason in scores["technical_reasons"][:6]:
        print(f"    - {reason}")
    print("  펀더멘털 근거:")
    for reason in scores["fundamental_reasons"][:6]:
        print(f"    - {reason}")
    print()



def _print_recent_ohlc(report: dict[str, Any]):
    recent_ohlc = report["recent_ohlc"]
    if recent_ohlc is None:
        return
    print("[최근 5일 OHLC]")
    print(recent_ohlc.to_string())



def run_analysis(ticker: str, period: str = "2y", show_chart_data: bool = True, term: Term = "all") -> dict[str, Any]:
    try:
        report = build_analysis_report(
            ticker=ticker,
            period=period,
            show_chart_data=show_chart_data,
            term=term,
        )
    except Exception as exc:
        print(f"데이터 조회 실패: {exc}")
        return {"ticker": ticker.upper(), "error": str(exc)}

    _print_header(report)
    _print_warnings(report)
    _print_base_info(report)
    _print_sector_info(report)
    _print_price_summary(report)
    _print_sections(report)
    _print_scores(report)
    _print_recent_ohlc(report)

    return report


