from typing import Literal, Optional

import pandas as pd

from .config import RELATIVE_PERIOD_LABELS, TERM_CONFIG
from .pivot import get_support_resistance_pivot
from .utils import fmt_big_num, fmt_num


SECTOR_NAME_MAP = {
    "Technology": "기술",
    "Healthcare": "헬스케어",
    "Financial Services": "금융 서비스",
    "Financials": "금융",
    "Consumer Cyclical": "경기소비재",
    "Consumer Defensive": "필수소비재",
    "Energy": "에너지",
    "Industrials": "산업재",
    "Basic Materials": "소재",
    "Real Estate": "부동산",
    "Communication Services": "커뮤니케이션 서비스",
    "Utilities": "유틸리티",
}

SECTOR_ETF_NAME_MAP = {
    "XLK": "기술주 대표 ETF",
    "XLV": "헬스케어 대표 ETF",
    "XLF": "금융 대표 ETF",
    "XLY": "경기소비재 대표 ETF",
    "XLP": "필수소비재 대표 ETF",
    "XLE": "에너지 대표 ETF",
    "XLI": "산업재 대표 ETF",
    "XLB": "소재 대표 ETF",
    "XLRE": "부동산 대표 ETF",
    "XLC": "커뮤니케이션 서비스 대표 ETF",
    "XLU": "유틸리티 대표 ETF",
}

INDUSTRY_NAME_MAP = {
    "Consumer Electronics": "소비자 전자기기",
    "Semiconductors": "반도체",
    "Software - Infrastructure": "인프라 소프트웨어",
    "Software - Application": "응용 소프트웨어",
    "Internet Content & Information": "인터넷 콘텐츠 및 정보",
    "Communication Equipment": "통신 장비",
    "Medical Devices": "의료기기",
    "Drug Manufacturers - General": "제약 일반",
    "Banks - Diversified": "종합은행",
    "Credit Services": "신용 서비스",
    "Oil & Gas Integrated": "종합 석유·가스",
    "Aerospace & Defense": "항공우주·방산",
    "Auto Manufacturers": "자동차 제조",
    "Internet Retail": "인터넷 소매",
}


def _format_sector_name(sector: str) -> str:
    if not sector or sector == "-":
        return "-"
    return SECTOR_NAME_MAP.get(sector, sector)



def _format_sector_etf(etf: str) -> str:
    label = SECTOR_ETF_NAME_MAP.get(etf)
    return f"{label} ({etf})" if label else etf


def _format_industry_name(industry: str) -> str:
    if not industry or industry == "-":
        return "-"
    return INDUSTRY_NAME_MAP.get(industry, industry)


def analyze_sector_comparison(comp: dict) -> list[str]:
    lines = []

    sector = _format_sector_name(comp.get("sector", "-"))
    industry = _format_industry_name(comp.get("industry", "-"))
    sector_etf = comp.get("sector_etf")

    lines.append(f"섹터: {sector} / 산업: {industry}")
    if sector_etf:
        lines.append(f"섹터 대표 ETF: {_format_sector_etf(sector_etf)}")

    has_relative = False
    for key, label in RELATIVE_PERIOD_LABELS:
        t_ret = comp.get(f"ticker_return_{key}")
        s_ret = comp.get(f"sector_return_{key}")
        diff = comp.get(f"return_vs_sector_{key}")
        if t_ret is None or s_ret is None:
            continue
        has_relative = True
        lines.append(f"{label} 수익률: 종목 {t_ret:+.1f}% / 섹터 평균 {s_ret:+.1f}%")
        if diff is None:
            continue
        if diff >= 10:
            lines.append(f"→ {label} 기준 섹터 대비 강한 아웃퍼폼 ({diff:+.1f}%p)")
        elif diff >= 3:
            lines.append(f"→ {label} 기준 섹터 대비 아웃퍼폼 ({diff:+.1f}%p)")
        elif diff <= -10:
            lines.append(f"→ {label} 기준 섹터 대비 강한 언더퍼폼 ({diff:+.1f}%p)")
        elif diff <= -3:
            lines.append(f"→ {label} 기준 섹터 대비 언더퍼폼 ({diff:+.1f}%p)")
        else:
            lines.append(f"→ {label} 기준 섹터와 유사한 흐름 ({diff:+.1f}%p)")

    if not has_relative:
        lines.append("상대강도 데이터 부족")

    target_per = comp.get("target_per")
    sector_per = comp.get("sector_per")
    if target_per is not None and sector_per is not None:
        per_diff_pct = (target_per - sector_per) / sector_per * 100
        lines.append(f"PER: 종목 {target_per:.1f}x / 섹터 대표 ETF {sector_per:.1f}x")
        if per_diff_pct >= 30:
            lines.append(f"→ 섹터 대비 PER {per_diff_pct:+.0f}% 프리미엄 (고평가 주의)")
        elif per_diff_pct >= 10:
            lines.append(f"→ 섹터 대비 PER {per_diff_pct:+.0f}% 프리미엄")
        elif per_diff_pct <= -30:
            lines.append(f"→ 섹터 대비 PER {per_diff_pct:+.0f}% 할인 (저평가 가능성)")
        elif per_diff_pct <= -10:
            lines.append(f"→ 섹터 대비 PER {per_diff_pct:+.0f}% 할인")
        else:
            lines.append(f"→ 섹터와 유사한 PER 수준 ({per_diff_pct:+.0f}%)")
    elif target_per is not None:
        lines.append(f"PER: {target_per:.1f}x (섹터 비교 불가)")

    fpe = comp.get("target_forward_pe")
    if fpe is not None:
        lines.append(f"선행 PER: {fpe:.1f}x")

    target_margin = comp.get("target_op_margin")
    if target_margin is not None:
        lines.append(f"영업이익률: {target_margin:.1f}%")

    target_roe = comp.get("target_roe")
    if target_roe is not None:
        lines.append(f"ROE: {target_roe:.1f}%")

    weight = comp.get("market_weight_pct")
    if weight is not None:
        lines.append(f"산업 내 시가총액 비중: {weight:.2f}%")

    return lines or ["섹터 비교 데이터 없음"]


def analyze_trend(
    df: pd.DataFrame,
    term: Literal["short", "medium", "long"],
    current_price: Optional[float] = None,
) -> list[str]:
    cfg = TERM_CONFIG[term]
    fast, slow = cfg["ma_fast"], cfg["ma_slow"]
    latest = df.iloc[-1]

    v_fast = latest.get(f"MA_{fast}")
    v_slow = latest.get(f"MA_{slow}")

    if pd.isna(v_fast) or pd.isna(v_slow):
        return [f"데이터 부족 (최소 {slow}일 필요)"]

    price = float(current_price) if current_price is not None else float(latest["Close"])
    return [
        f"종가/현재가가 {slow}일선 {'위' if price > v_slow else '아래'} → {'상승' if price > v_slow else '하락'} 추세",
        f"{fast}일선 {'위' if price > v_fast else '아래'}",
        f"{'정배열' if v_fast > v_slow else '역배열'} ({fast}일 {'>' if v_fast > v_slow else '<'} {slow}일)",
    ]


def analyze_volume(
    df: pd.DataFrame,
    term: Literal["short", "medium", "long"],
) -> list[str]:
    n = TERM_CONFIG[term]["volume_days"]
    if len(df) < n * 2:
        return [f"데이터 부족 (최소 {n*2}일)"]

    recent = df["Volume"].tail(n).mean()
    prior = df["Volume"].tail(n * 2).head(n).mean()
    if prior <= 0 or pd.isna(prior):
        return ["거래량 비교 불가"]

    pct = (recent - prior) / prior * 100
    lines = [
        f"최근 {n}일 평균 거래량: {fmt_big_num(recent)}",
        f"이전 {n}일 대비: {pct:+.1f}%",
    ]
    if pct > 20:
        lines.append("→ 거래량 증가")
    elif pct < -20:
        lines.append("→ 거래량 감소")
    return lines


def analyze_speed(
    df: pd.DataFrame,
    term: Literal["short", "medium", "long"],
) -> list[str]:
    if len(df) < 30:
        return ["데이터 부족"]

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    cfg = TERM_CONFIG[term]
    lines = []

    rsi = latest.get("RSI")
    if rsi is not None and not pd.isna(rsi):
        if rsi > cfg["rsi_neutral_high"] + 5:
            lines.append(f"RSI {rsi:.0f} → 과매수")
        elif rsi < max(30, cfg["rsi_neutral_low"] - 15):
            lines.append(f"RSI {rsi:.0f} → 과매도")
        else:
            lines.append(f"RSI {rsi:.0f} → 중립")

    hist = latest.get("MACD_hist")
    prev_hist = prev.get("MACD_hist")
    if hist is not None and not pd.isna(hist):
        lines.append(f"MACD 히스토그램: {hist:.4f}")
        if prev_hist is not None and not pd.isna(prev_hist):
            if hist > 0 and prev_hist <= 0:
                lines.append("→ MACD 골든크로스")
            elif hist < 0 and prev_hist >= 0:
                lines.append("→ MACD 데드크로스")
            elif hist > prev_hist:
                lines.append("→ MACD 모멘텀 개선")
            elif hist < prev_hist:
                lines.append("→ MACD 모멘텀 둔화")

    return lines or ["RSI/MACD 계산 불가"]


def analyze_volatility(
    df: pd.DataFrame,
    term: Literal["short", "medium", "long"],
    current_price: Optional[float] = None,
) -> list[str]:
    if len(df) < 20:
        return ["데이터 부족 (20일 이상)"]

    latest = df.iloc[-1]
    price = float(current_price) if current_price is not None else float(latest["Close"])
    upper = latest["BB_upper"]
    lower = latest["BB_lower"]
    mid = latest["BB_mid"]
    width = latest["BB_width_pct"]

    if pd.isna(upper) or pd.isna(lower) or pd.isna(mid):
        return ["볼린저밴드 계산 불가"]

    lines = [f"상단: {upper:.2f} / 중심: {mid:.2f} / 하단: {lower:.2f}"]
    if price >= upper:
        lines.append("→ 상단 밴드 근처/돌파")
    elif price <= lower:
        lines.append("→ 하단 밴드 근처/이탈")
    else:
        pct_in = (price - lower) / (upper - lower) * 100 if upper != lower else 50
        lines.append(f"→ 밴드 내 {pct_in:.0f}% 구간")
    lines.append(f"밴드폭: {width:.2f}%")
    return lines


def analyze_support_resistance(
    df: pd.DataFrame,
    term: Literal["short", "medium", "long"],
    current_price: Optional[float] = None,
) -> list[str]:
    lookback = TERM_CONFIG[term]["sr_days"]
    price = float(current_price) if current_price is not None else float(df.iloc[-1]["Close"])

    support, resistance, support_levels, resistance_levels = get_support_resistance_pivot(
        df, lookback_days=lookback, current_price=price, left=3, right=3, tolerance_pct=1.5
    )

    lines = [f"기준: 최근 {lookback}일 pivot 기반 지지/저항"]

    if support is not None:
        to_sup = (price - support) / price * 100
        lines.append(f"가장 가까운 지지: {support:.2f} (현재가 대비 {to_sup:.1f}% 아래)")
        if price <= support * 1.02:
            lines.append("→ 지지 근처")
    else:
        lines.append("가장 가까운 지지: 없음")

    if resistance is not None:
        to_res = (resistance - price) / price * 100
        lines.append(f"가장 가까운 저항: {resistance:.2f} (현재가 대비 {to_res:.1f}% 위)")
        if price >= resistance * 0.98:
            lines.append("→ 저항 근처")
    else:
        lines.append("가장 가까운 저항: 없음")

    if support_levels:
        lines.append("주요 지지 후보: " + ", ".join(f"{x:.2f}" for x in support_levels[-3:]))
    if resistance_levels:
        lines.append("주요 저항 후보: " + ", ".join(f"{x:.2f}" for x in resistance_levels[:3]))

    return lines


def analyze_long_term_fundamentals(
    df: pd.DataFrame,
    fin: dict,
    current_price: Optional[float] = None,
) -> list[str]:
    latest = df.iloc[-1]
    price = float(current_price) if current_price is not None else float(latest["Close"])
    high_52w = latest.get("HIGH_52W")
    low_52w = latest.get("LOW_52W")

    lines = []

    # --- 52주 위치 ---
    if high_52w is not None and low_52w is not None and not pd.isna(high_52w) and not pd.isna(low_52w):
        pos = (price - low_52w) / (high_52w - low_52w) * 100 if high_52w != low_52w else 50
        lines.append(f"52주 고점: {high_52w:.2f} / 52주 저점: {low_52w:.2f}")
        lines.append(f"현재가는 52주 범위의 {pos:.1f}% 위치")

    # --- 수익성 ---
    lines.append("")
    lines.append("[수익성]")
    lines.append(f"매출총이익률: {fmt_num(fin.get('gross_margin_pct'))}%")
    lines.append(f"영업이익률: {fmt_num(fin.get('operating_margin_pct'))}%")
    lines.append(f"순이익률: {fmt_num(fin.get('net_margin_pct'))}%")
    lines.append(f"ROE: {fmt_num(fin.get('roe_pct'))}%")
    lines.append(f"ROA: {fmt_num(fin.get('roa_pct'))}%")

    payout = fin.get("payout_ratio_pct")
    if payout is not None:
        lines.append(f"배당성향: {fmt_num(payout)}%")
        if payout > 100:
            lines.append("→ 배당성향 과다 (이익 초과 배당)")
        elif 20 <= payout <= 60:
            lines.append("→ 배당성향 적정 범위")

    # --- 성장성 ---
    lines.append("")
    lines.append("[성장성]")
    lines.append(f"연간 매출 성장률: {fmt_num(fin.get('annual_revenue_yoy_pct'))}%")
    lines.append(f"분기 매출 성장률: {fmt_num(fin.get('quarterly_revenue_yoy_pct'))}%")
    lines.append(f"연간 순이익 성장률: {fmt_num(fin.get('annual_net_income_yoy_pct'))}%")
    lines.append(f"분기 순이익 성장률: {fmt_num(fin.get('quarterly_net_income_yoy_pct'))}%")
    lines.append(f"영업현금흐름 성장률: {fmt_num(fin.get('ocf_yoy_pct'))}%")

    # --- 현금흐름 ---
    lines.append("")
    lines.append("[현금흐름]")
    lines.append(f"영업현금흐름: {fmt_big_num(fin.get('operating_cash_flow'))}")
    lines.append(f"잉여현금흐름: {fmt_big_num(fin.get('free_cash_flow'))}")
    lines.append(f"이익의 질(OCF/NI): {fmt_num(fin.get('earnings_quality_ratio'))}")
    lines.append(f"FCF 마진: {fmt_num(fin.get('fcf_margin_pct'))}%")
    lines.append(f"CAPEX/매출: {fmt_num(fin.get('capex_to_revenue_pct'))}%")

    fcf = fin.get("free_cash_flow")
    if fcf is not None:
        lines.append("→ 잉여현금흐름 " + ("양호(+)" if fcf > 0 else "주의(-)"))

    fcf_margin = fin.get("fcf_margin_pct")
    if fcf_margin is not None:
        if fcf_margin >= 15:
            lines.append("→ FCF 마진 우수 (매출의 현금 전환 효율 높음)")
        elif fcf_margin < 0:
            lines.append("→ FCF 마진 음수 (현금 유출 상태)")

    # --- 재무 안정성 ---
    lines.append("")
    lines.append("[재무 안정성]")
    lines.append(f"부채비율(D/E): {fmt_num(fin.get('debt_to_equity_pct'))}%")
    lines.append(f"유동비율: {fmt_num(fin.get('current_ratio'))}")
    lines.append(f"당좌비율: {fmt_num(fin.get('quick_ratio'))}")
    lines.append(f"이자보상배율: {fmt_num(fin.get('interest_coverage'))}x")
    lines.append(f"순부채/EBITDA: {fmt_num(fin.get('net_debt_to_ebitda'))}x")
    lines.append(f"주식수 증감률: {fmt_num(fin.get('share_change_yoy_pct'))}%")

    de = fin.get("debt_to_equity_pct")
    if de is not None:
        if de < 100:
            lines.append("→ 부채비율 안정권")
        elif de < 200:
            lines.append("→ 부채비율 보통")
        else:
            lines.append("→ 부채 부담 주의")

    interest_cov = fin.get("interest_coverage")
    if interest_cov is not None:
        if interest_cov < 1.5:
            lines.append("→ 이자보상배율 위험 (이자 지급 능력 부족)")
        elif interest_cov >= 5:
            lines.append("→ 이자보상배율 양호")

    nde = fin.get("net_debt_to_ebitda")
    if nde is not None:
        if nde > 5:
            lines.append("→ 순부채/EBITDA 과다 (부채 상환 부담 높음)")
        elif nde < 1:
            lines.append("→ 순부채/EBITDA 우수 (부채 부담 매우 낮음)")

    # --- 밸류에이션 ---
    lines.append("")
    lines.append("[밸류에이션]")
    lines.append(f"EV/EBITDA: {fmt_num(fin.get('ev_to_ebitda'))}x")
    lines.append(f"PEG Ratio: {fmt_num(fin.get('peg_ratio'))}")
    lines.append(f"P/S (주가매출비율): {fmt_num(fin.get('price_to_sales'))}x")
    lines.append(f"P/FCF (주가잉여현금흐름비율): {fmt_num(fin.get('price_to_fcf'))}x")
    lines.append(f"FCF Yield: {fmt_num(fin.get('fcf_yield_pct'))}%")

    ev_ebitda = fin.get("ev_to_ebitda")
    if ev_ebitda is not None:
        if ev_ebitda > 30:
            lines.append("→ EV/EBITDA 고평가 구간")
        elif ev_ebitda < 10:
            lines.append("→ EV/EBITDA 저평가 가능성")

    peg = fin.get("peg_ratio")
    if peg is not None:
        if 0 < peg <= 1.0:
            lines.append("→ PEG 기준 성장 대비 저평가")
        elif peg > 2.0:
            lines.append("→ PEG 기준 성장 대비 고평가")

    return lines
