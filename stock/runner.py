from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "분석자료"


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
            items.append(
                {
                    "name": section_names["fundamentals"],
                    "lines": analyze_long_term_fundamentals(df, fin, current_price=current_price),
                }
            )
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


def _append_blank(lines: list[str]):
    if lines and lines[-1] != "":
        lines.append("")


def _render_report_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"{report['ticker']} 투자 분석 (Yahoo Finance)")
    lines.append("=" * 60)
    _append_blank(lines)

    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"[참고] {warning}")
        _append_blank(lines)

    lines.append("[기본 정보]")
    for key, value in report["base_info"].items():
        if value is not None and value != "-":
            lines.append(f"{key}: {value}")
    _append_blank(lines)

    lines.append("[섹터/산업 대비 분석]")
    for line in report["sector_lines"]:
        lines.append(line)
    _append_blank(lines)

    summary = report["price_summary"]
    latest_date = summary["latest_date"]
    lines.append("[최근 가격]")
    lines.append(f"일봉 기준 일자: {latest_date.date() if hasattr(latest_date, 'date') else latest_date}")
    lines.append(f"정규장 종가: {summary['daily_close']:.2f}")
    lines.append(f"분석 기준 가격: {summary['current_price']:.2f} ({summary['market_state']})")
    if summary["ext_time"] is not None:
        lines.append(f"연장장 체결 시각: {summary['ext_time']}")
    if summary["diff_pct"] is not None:
        lines.append(f"정규장 종가 대비: {summary['diff_pct']:+.2f}%")
    for window, value in summary["moving_averages"].items():
        lines.append(f"MA({window}): {value:.2f}")
    if summary["rsi"] is not None:
        lines.append(f"RSI: {summary['rsi']:.1f}")
    _append_blank(lines)

    for section in report["sections"]:
        lines.append(f"[{section['label']} 분석]")
        for item in section["items"]:
            lines.append(f"- {item['name']}")
            for line in item["lines"]:
                lines.append(f"  - {line}")
        _append_blank(lines)

    scores = report["scores"]
    lines.append("[매수/매도 신호 점수]")
    lines.append(f"점수 기준 기간: {scores['term']}")
    lines.append(f"기술적 점수: {scores['technical']}/100")
    lines.append(f"펀더멘털 점수: {scores['fundamentals']}/100")
    lines.append(f"종합 점수: {scores['total']}/100 -> {scores['label']}")
    lines.append("기술 팩터:")
    for name, value in scores["technical_factors"].items():
        lines.append(f"- {name}: {value}")
    lines.append("재무 팩터:")
    for name, value in scores["fundamental_factors"].items():
        lines.append(f"- {name}: {value}")
    lines.append("기술적 근거:")
    for reason in scores["technical_reasons"][:6]:
        lines.append(f"- {reason}")
    lines.append("펀더멘털 근거:")
    for reason in scores["fundamental_reasons"][:6]:
        lines.append(f"- {reason}")

    recent_ohlc = report["recent_ohlc"]
    if recent_ohlc is not None:
        _append_blank(lines)
        lines.append("[최근 5일 OHLC]")
        lines.extend(recent_ohlc.to_string().splitlines())

    return lines


def _safe_filename_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return cleaned.strip("._") or "report"


def _get_pdf_font_properties() -> Optional[font_manager.FontProperties]:
    font_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/gulim.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return font_manager.FontProperties(fname=str(font_path))
    return None


def _save_report_pdf(report: dict[str, Any]) -> Path:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ticker = _safe_filename_part(report["ticker"])
    output_path = ANALYSIS_OUTPUT_DIR / f"{timestamp}_{ticker}.pdf"
    font_prop = _get_pdf_font_properties()

    def add_text(fig, x: float, y: float, text: str, size: int = 10, weight: str = "normal", color: str = "#111827"):
        fig.text(
            x,
            y,
            text,
            ha="left",
            va="top",
            fontsize=size,
            fontproperties=font_prop,
            family=None if font_prop else "DejaVu Sans",
            weight=weight,
            color=color,
        )

    def draw_card(fig, y_top: float, height: float, title: str) -> float:
        rect = Rectangle((0.05, y_top - height), 0.90, height, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.8)
        fig.add_artist(rect)
        add_text(fig, 0.07, y_top - 0.02, title, size=12, weight="bold")
        return y_top - 0.055

    summary = report["price_summary"]
    scores = report["scores"]

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        plt.axis("off")

        add_text(fig, 0.05, 0.965, f"{report['ticker']} 투자 분석 리포트", size=18, weight="bold")
        add_text(fig, 0.05, 0.935, f"생성시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}", size=9, color="#475569")
        add_text(fig, 0.62, 0.935, f"분석기간: {scores['term']} / 조회기간: {report['period']}", size=9, color="#475569")

        body_y = draw_card(fig, 0.89, 0.14, "핵심 요약")
        add_text(fig, 0.07, body_y, f"종합 점수: {scores['total']}/100", size=13, weight="bold")
        add_text(fig, 0.34, body_y, f"판정: {scores['label']}", size=13, weight="bold", color="#0F766E")
        add_text(fig, 0.60, body_y, f"기술: {scores['technical']}/100", size=11)
        add_text(fig, 0.78, body_y, f"재무: {scores['fundamentals']}/100", size=11)
        add_text(fig, 0.07, body_y - 0.045, f"기준 가격: {summary['current_price']:.2f} ({summary['market_state']})", size=10)
        add_text(fig, 0.40, body_y - 0.045, f"정규장 종가: {summary['daily_close']:.2f}", size=10)
        if summary["diff_pct"] is not None:
            add_text(fig, 0.67, body_y - 0.045, f"종가 대비: {summary['diff_pct']:+.2f}%", size=10)

        body_y = draw_card(fig, 0.72, 0.18, "기술 / 재무 팩터")
        add_text(fig, 0.07, body_y, "기술 팩터", size=11, weight="bold")
        for idx, (name, value) in enumerate(scores["technical_factors"].items()):
            add_text(fig, 0.07, body_y - 0.03 - idx * 0.028, f"{name}: {value}", size=10)
        add_text(fig, 0.52, body_y, "재무 팩터", size=11, weight="bold")
        for idx, (name, value) in enumerate(scores["fundamental_factors"].items()):
            add_text(fig, 0.52, body_y - 0.03 - idx * 0.028, f"{name}: {value}", size=10)

        body_y = draw_card(fig, 0.50, 0.18, "상대강도 / 최근 가격")
        for idx, line in enumerate(report["sector_lines"][:6]):
            add_text(fig, 0.07, body_y - idx * 0.028, line, size=9)
        price_x = 0.55
        add_text(fig, price_x, body_y, "주요 가격 지표", size=11, weight="bold")
        latest_date = summary["latest_date"]
        add_text(fig, price_x, body_y - 0.03, f"일자: {latest_date.date() if hasattr(latest_date, 'date') else latest_date}", size=9)
        add_text(fig, price_x, body_y - 0.058, f"RSI: {summary['rsi']:.1f}" if summary["rsi"] is not None else "RSI: -", size=9)
        for idx, (window, value) in enumerate(list(summary["moving_averages"].items())[:4]):
            add_text(fig, price_x, body_y - 0.086 - idx * 0.028, f"MA({window}): {value:.2f}", size=9)

        body_y = draw_card(fig, 0.28, 0.20, "주요 근거")
        add_text(fig, 0.07, body_y, "기술적 근거", size=11, weight="bold")
        for idx, reason in enumerate(scores["technical_reasons"][:5]):
            add_text(fig, 0.07, body_y - 0.03 - idx * 0.026, f"- {reason}", size=9)
        add_text(fig, 0.52, body_y, "펀더멘털 근거", size=11, weight="bold")
        for idx, reason in enumerate(scores["fundamental_reasons"][:5]):
            add_text(fig, 0.52, body_y - 0.03 - idx * 0.026, f"- {reason}", size=9)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        report_lines = _render_report_lines(report)
        lines_per_page = 44
        line_height = 0.020
        for start in range(0, len(report_lines), lines_per_page):
            page_lines = report_lines[start:start + lines_per_page]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            plt.axis("off")
            add_text(fig, 0.05, 0.965, f"{report['ticker']} 상세 리포트", size=15, weight="bold")

            y = 0.93
            for line in page_lines:
                add_text(fig, 0.05, y, line, size=9)
                y -= line_height

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output_path


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

    pdf_output_path = _save_report_pdf(report)
    print(f"[저장 완료] PDF: {pdf_output_path}")

    report["saved_pdf_path"] = str(pdf_output_path)
    return report
