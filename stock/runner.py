import textwrap
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
    _format_industry_name,
    _format_sector_name,
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

MARKET_STATE_LABELS = {
    "REGULAR_CLOSE": "정규장 종가 기준",
    "PREMARKET": "프리마켓",
    "AFTERMARKET": "애프터마켓",
    "EXTENDED": "시간외 거래",
}

FACTOR_LABELS = {
    "trend": "추세",
    "momentum": "모멘텀",
    "volatility": "변동성",
    "flow": "수급",
    "relative": "상대강도",
    "growth": "성장성",
    "quality": "현금흐름/이익의 질",
    "balance": "재무안정성",
    "profitability": "수익성",
    "market_position": "시장 위치",
}

TERM_LABELS = {
    "short": "단기",
    "medium": "중기",
    "long": "장기",
    "all": "전체",
}

PERIOD_SUFFIX_LABELS = {
    "d": "일",
    "mo": "개월",
    "y": "년",
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


def _format_term_label(term: str) -> str:
    return TERM_LABELS.get(term, term)


def _format_period_label(period: str) -> str:
    if period == "max":
        return "전체 기간"
    for suffix, label in PERIOD_SUFFIX_LABELS.items():
        if period.endswith(suffix):
            value = period[: -len(suffix)]
            if value.isdigit():
                return f"{int(value)}{label}"
    return period


def _format_base_info_value(key: str, value: Any) -> Any:
    if key == "섹터":
        return _format_sector_name(str(value))
    if key == "산업":
        return _format_industry_name(str(value))
    return value


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


def _collect_common_section_lines(sections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(sections) < 2:
        return [], sections

    common_items: list[dict[str, Any]] = []
    section_item_maps = [{item["name"]: item for item in section["items"]} for section in sections]
    common_names = [
        item["name"]
        for item in sections[0]["items"]
        if all(item["name"] in section_map for section_map in section_item_maps[1:])
    ]

    for item_name in common_names:
        line_lists = [section_map[item_name]["lines"] for section_map in section_item_maps]
        common_lines = [line for line in line_lists[0] if line and all(line in lines for lines in line_lists[1:])]
        if common_lines:
            common_items.append({"name": item_name, "lines": common_lines})

    if not common_items:
        return [], sections

    filtered_sections: list[dict[str, Any]] = []
    for section in sections:
        filtered_items = []
        for item in section["items"]:
            common_lines = next((common["lines"] for common in common_items if common["name"] == item["name"]), [])
            unique_lines = [line for line in item["lines"] if line not in common_lines]
            if unique_lines:
                filtered_items.append({**item, "lines": unique_lines})
        if filtered_items:
            filtered_sections.append({**section, "items": filtered_items})

    return common_items, filtered_sections


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

    if term == "all":
        common_items, filtered_sections = _collect_common_section_lines(sections)
        if common_items:
            return [{"term": "summary", "label": "종합 자료", "items": common_items}, *filtered_sections]

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
    company_name = report["base_info"].get("종목명") or report["ticker"]
    print(f"\n{'='*60}")
    print(f"  {company_name} ({report['ticker']}) 투자 분석")
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
            print(f"  {key}: {_format_base_info_value(key, value)}")
    print()


def _print_sector_info(report: dict[str, Any]):
    print("[섹터/산업 대비 분석]")
    for line in report["sector_lines"]:
        print(f"  {line}")
    print()


def _print_price_summary(report: dict[str, Any]):
    summary = report["price_summary"]
    market_state_label = MARKET_STATE_LABELS.get(summary["market_state"], summary["market_state"])

    print("[최근 가격]")
    latest_date = summary["latest_date"]
    print(f"  일봉 기준 일자: {latest_date.date() if hasattr(latest_date, 'date') else latest_date}")
    print(f"  정규장 종가: {summary['daily_close']:.2f}")
    print(f"  분석 기준 가격: {summary['current_price']:.2f} ({market_state_label})")

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
    print(f"  점수 기준 기간: {_format_term_label(scores['term'])}")
    print(f"  기술적 점수: {scores['technical']}/100")
    print(f"  펀더멘털 점수: {scores['fundamentals']}/100")
    print(f"  종합 점수: {scores['total']}/100 → {scores['label']}")
    print("  기술 팩터:")
    for name, value in scores["technical_factors"].items():
        print(f"    - {FACTOR_LABELS.get(name, name)}: {value}")
    print("  재무 팩터:")
    for name, value in scores["fundamental_factors"].items():
        print(f"    - {FACTOR_LABELS.get(name, name)}: {value}")
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
    company_name = report["base_info"].get("종목명") or report["ticker"]
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"{company_name} ({report['ticker']}) 투자 분석")
    lines.append("=" * 60)
    _append_blank(lines)

    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"[참고] {warning}")
        _append_blank(lines)

    lines.append("[기본 정보]")
    for key, value in report["base_info"].items():
        if value is not None and value != "-":
            lines.append(f"{key}: {_format_base_info_value(key, value)}")
    _append_blank(lines)

    lines.append("[섹터/산업 대비 분석]")
    for line in report["sector_lines"]:
        lines.append(line)
    _append_blank(lines)

    summary = report["price_summary"]
    latest_date = summary["latest_date"]
    market_state_label = MARKET_STATE_LABELS.get(summary["market_state"], summary["market_state"])
    lines.append("[최근 가격]")
    lines.append(f"일봉 기준 일자: {latest_date.date() if hasattr(latest_date, 'date') else latest_date}")
    lines.append(f"정규장 종가: {summary['daily_close']:.2f}")
    lines.append(f"분석 기준 가격: {summary['current_price']:.2f} ({market_state_label})")
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
    lines.append(f"점수 기준 기간: {_format_term_label(scores['term'])}")
    lines.append(f"기술적 점수: {scores['technical']}/100")
    lines.append(f"펀더멘털 점수: {scores['fundamentals']}/100")
    lines.append(f"종합 점수: {scores['total']}/100 -> {scores['label']}")
    lines.append("기술 팩터:")
    for name, value in scores["technical_factors"].items():
        lines.append(f"- {FACTOR_LABELS.get(name, name)}: {value}")
    lines.append("재무 팩터:")
    for name, value in scores["fundamental_factors"].items():
        lines.append(f"- {FACTOR_LABELS.get(name, name)}: {value}")
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


def _wrap_pdf_text(text: str, width: int) -> list[str]:
    raw_lines = str(text).splitlines() or [""]
    wrapped: list[str] = []
    for raw_line in raw_lines:
        chunks = textwrap.wrap(
            raw_line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped.extend(chunks or [""])
    return wrapped


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

    def draw_text_column(fig, x: float, y_top: float, lines: list[str], *, width: int, size: int = 10, line_height: float = 0.024):
        y = y_top
        for line in lines:
            for wrapped_line in _wrap_pdf_text(line, width):
                add_text(fig, x, y, wrapped_line, size=size)
                y -= line_height
        return y

    def draw_two_column_card(
        fig,
        y_top: float,
        title: str,
        left_title: str,
        left_lines: list[str],
        right_title: str,
        right_lines: list[str],
        *,
        left_width: int = 34,
        right_width: int = 30,
        size: int = 9,
        line_height: float = 0.022,
    ) -> float:
        left_wrapped = [wrapped for line in left_lines for wrapped in _wrap_pdf_text(line, left_width)]
        right_wrapped = [wrapped for line in right_lines for wrapped in _wrap_pdf_text(line, right_width)]
        row_count = max(len(left_wrapped), len(right_wrapped), 1)
        height = 0.085 + row_count * line_height
        body_y = draw_card(fig, y_top, height, title)
        add_text(fig, 0.07, body_y, left_title, size=11, weight="bold")
        add_text(fig, 0.52, body_y, right_title, size=11, weight="bold")
        draw_text_column(fig, 0.07, body_y - 0.03, left_lines, width=left_width, size=size, line_height=line_height)
        draw_text_column(fig, 0.52, body_y - 0.03, right_lines, width=right_width, size=size, line_height=line_height)
        return y_top - height - 0.025

    def draw_full_width_card(
        fig,
        y_top: float,
        title: str,
        lines: list[str],
        *,
        width: int = 78,
        size: int = 9,
        line_height: float = 0.022,
    ) -> float:
        wrapped = [wrapped for line in lines for wrapped in _wrap_pdf_text(line, width)]
        height = 0.085 + max(len(wrapped), 1) * line_height
        body_y = draw_card(fig, y_top, height, title)
        draw_text_column(fig, 0.07, body_y, lines, width=width, size=size, line_height=line_height)
        return y_top - height - 0.025

    summary = report["price_summary"]
    scores = report["scores"]

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        plt.axis("off")

        company_name = report["base_info"].get("종목명") or report["ticker"]
        add_text(fig, 0.05, 0.965, f"{company_name} ({report['ticker']}) 투자 분석 리포트", size=18, weight="bold")
        add_text(fig, 0.05, 0.935, f"생성시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}", size=9, color="#475569")
        add_text(fig, 0.62, 0.935, f"분석기간: {_format_term_label(scores['term'])} / 조회기간: {_format_period_label(report['period'])}", size=9, color="#475569")

        diff_label = f"종가 대비: {summary['diff_pct']:+.2f}%" if summary["diff_pct"] is not None else "종가 대비: -"
        latest_date = summary["latest_date"]
        key_summary_lines = [
            f"종합 점수: {scores['total']}/100",
            f"판정: {scores['label']}",
            f"기술 점수: {scores['technical']}/100",
            f"재무 점수: {scores['fundamentals']}/100",
        ]
        quick_view_lines = [
            f"종목: {company_name} ({report['ticker']})",
            f"분석기간: {_format_term_label(scores['term'])}",
            f"조회기간: {_format_period_label(report['period'])}",
            f"기준 상태: {MARKET_STATE_LABELS.get(summary['market_state'], summary['market_state'])}",
        ]
        price_lines = [
            f"기준 가격: {summary['current_price']:.2f} ({MARKET_STATE_LABELS.get(summary['market_state'], summary['market_state'])})",
            f"정규장 종가: {summary['daily_close']:.2f}",
            diff_label,
            f"일자: {latest_date.date() if hasattr(latest_date, 'date') else latest_date}",
            f"RSI: {summary['rsi']:.1f}" if summary["rsi"] is not None else "RSI: -",
        ]
        for window, value in list(summary["moving_averages"].items())[:4]:
            price_lines.append(f"MA({window}): {value:.2f}")

        y_cursor = 0.89
        y_cursor = draw_two_column_card(
            fig,
            y_cursor,
            "핵심 요약",
            "점수 요약",
            key_summary_lines,
            "조회 정보",
            quick_view_lines,
            left_width=28,
            right_width=30,
            size=10,
            line_height=0.024,
        )

        technical_factor_lines = [f"{FACTOR_LABELS.get(name, name)}: {value}" for name, value in scores["technical_factors"].items()]
        fundamental_factor_lines = [f"{FACTOR_LABELS.get(name, name)}: {value}" for name, value in scores["fundamental_factors"].items()]
        y_cursor = draw_two_column_card(
            fig,
            y_cursor,
            "기술 / 재무 팩터",
            "기술 팩터",
            technical_factor_lines,
            "재무 팩터",
            fundamental_factor_lines,
            left_width=24,
            right_width=24,
        )

        y_cursor = draw_two_column_card(
            fig,
            y_cursor,
            "상대강도 / 최근 가격",
            "섹터 비교",
            report["sector_lines"][:6],
            "주요 가격 지표",
            price_lines,
            left_width=34,
            right_width=26,
        )

        reason_lines = [f"기술: {reason}" for reason in scores["technical_reasons"][:4]]
        reason_lines.extend(f"재무: {reason}" for reason in scores["fundamental_reasons"][:4])
        draw_full_width_card(fig, y_cursor, "주요 근거", reason_lines, width=72)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        report_lines = _render_report_lines(report)
        wrapped_report_lines = [wrapped for line in report_lines for wrapped in _wrap_pdf_text(line, 78)]
        lines_per_page = 44
        line_height = 0.020
        for start in range(0, len(wrapped_report_lines), lines_per_page):
            page_lines = wrapped_report_lines[start:start + lines_per_page]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            plt.axis("off")
            add_text(fig, 0.05, 0.965, f"{company_name} ({report['ticker']}) 상세 리포트", size=15, weight="bold")

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
