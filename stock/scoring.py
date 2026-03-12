from typing import Optional

import pandas as pd

from .config import FUNDAMENTAL_SCORE_WEIGHTS, TERM_CONFIG


FactorMap = dict[str, int]


def _clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def _weighted_average(scores: dict[str, float], weights: dict[str, float]) -> int:
    total_weight = 0.0
    total_score = 0.0
    for name, weight in weights.items():
        if name not in scores:
            continue
        total_weight += weight
        total_score += scores[name] * weight
    if total_weight == 0:
        return 50
    return _clamp_score(total_score / total_weight)


def _score_relative_strength(comp: Optional[dict]) -> tuple[int, list[str]]:
    if not comp:
        return 50, ["상대강도 데이터 부족"]

    factors = []
    reasons = []
    for key, label in (("3m", "3개월"), ("6m", "6개월"), ("1y", "1년")):
        diff = comp.get(f"return_vs_sector_{key}")
        if diff is None and key == "1y":
            diff = comp.get("return_vs_sector")
        if diff is None:
            continue
        factors.append(max(-20.0, min(20.0, float(diff))) * 1.2)
        if diff >= 10:
            reasons.append(f"{label} 섹터 대비 강한 아웃퍼폼")
        elif diff >= 3:
            reasons.append(f"{label} 섹터 대비 아웃퍼폼")
        elif diff <= -10:
            reasons.append(f"{label} 섹터 대비 강한 언더퍼폼")
        elif diff <= -3:
            reasons.append(f"{label} 섹터 대비 언더퍼폼")

    if not factors:
        return 50, ["상대강도 데이터 부족"]

    score = _clamp_score(50 + sum(factors) / len(factors))
    return score, reasons or ["섹터 대비 수익률 중립"]


def score_technical(
    df: pd.DataFrame,
    current_price: Optional[float] = None,
    term: str = "medium",
    sector_comp: Optional[dict] = None,
) -> tuple[int, list[str], FactorMap]:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    cfg = TERM_CONFIG[term]

    price = float(current_price) if current_price is not None else float(latest["Close"])
    fast = cfg["ma_fast"]
    slow = cfg["ma_slow"]
    fast_ma = latest.get(f"MA_{fast}")
    slow_ma = latest.get(f"MA_{slow}")
    ma200 = latest.get("MA_200")
    rsi = latest.get("RSI")
    hist = latest.get("MACD_hist")
    prev_hist = prev.get("MACD_hist")
    bb_upper = latest.get("BB_upper")
    bb_lower = latest.get("BB_lower")
    adx = latest.get("ADX_14")
    obv = latest.get("OBV")
    obv_ma20 = latest.get("OBV_MA_20")
    atr_pct = latest.get("ATR_PCT")

    factor_scores: dict[str, float] = {}
    reasons: list[str] = []

    trend_score = 50.0
    if fast_ma is not None and not pd.isna(fast_ma):
        trend_score += 12 if price > fast_ma else -12
        reasons.append(f"현재가가 {fast}일선 {'상회' if price > fast_ma else '하회'}")
    if slow_ma is not None and not pd.isna(slow_ma):
        trend_score += 18 if price > slow_ma else -18
        reasons.append(f"현재가가 {slow}일선 {'상회' if price > slow_ma else '하회'}")
    if ma200 is not None and not pd.isna(ma200):
        trend_score += 10 if price > ma200 else -10
    if fast_ma is not None and slow_ma is not None and not pd.isna(fast_ma) and not pd.isna(slow_ma):
        trend_score += 8 if fast_ma > slow_ma else -8
    if adx is not None and not pd.isna(adx):
        if adx >= 25:
            trend_score += 6
            reasons.append("추세 강도 양호(ADX)")
        elif adx < 15:
            trend_score -= 4
    factor_scores["trend"] = _clamp_score(trend_score)

    momentum_score = 50.0
    neutral_low = cfg["rsi_neutral_low"]
    neutral_high = cfg["rsi_neutral_high"]
    if rsi is not None and not pd.isna(rsi):
        if neutral_low <= rsi <= neutral_high:
            momentum_score += 10
            reasons.append(f"RSI 중립 강세권({rsi:.1f})")
        elif rsi < 30:
            momentum_score += 4
            reasons.append(f"RSI 과매도 반등 구간({rsi:.1f})")
        elif rsi > 75:
            momentum_score -= 10
            reasons.append(f"RSI 과열 구간({rsi:.1f})")
        elif rsi < neutral_low:
            momentum_score -= 6
        else:
            momentum_score += 4
    if hist is not None and not pd.isna(hist):
        momentum_score += 10 if hist > 0 else -10
        if prev_hist is not None and not pd.isna(prev_hist):
            if hist > prev_hist:
                momentum_score += 5
                reasons.append("MACD 히스토그램 개선")
            elif hist < prev_hist:
                momentum_score -= 5
        if hist > 0 and prev_hist is not None and not pd.isna(prev_hist) and prev_hist <= 0:
            reasons.append("MACD 골든크로스")
        elif hist < 0 and prev_hist is not None and not pd.isna(prev_hist) and prev_hist >= 0:
            reasons.append("MACD 데드크로스")
    factor_scores["momentum"] = _clamp_score(momentum_score)

    volatility_score = 50.0
    if bb_upper is not None and bb_lower is not None and not pd.isna(bb_upper) and not pd.isna(bb_lower):
        if price >= bb_upper:
            volatility_score -= 10
            reasons.append("볼린저 상단 근접")
        elif price <= bb_lower:
            volatility_score += 6
            reasons.append("볼린저 하단 근접")
        else:
            band_pos = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            if 0.35 <= band_pos <= 0.75:
                volatility_score += 5
    if atr_pct is not None and not pd.isna(atr_pct):
        if atr_pct > cfg["atr_high_pct"]:
            volatility_score -= 8
            reasons.append(f"변동성 과다(ATR {atr_pct:.1f}%)")
        elif atr_pct < cfg["atr_high_pct"] * 0.6:
            volatility_score += 4
    factor_scores["volatility"] = _clamp_score(volatility_score)

    flow_score = 50.0
    n = cfg["volume_days"]
    if len(df) >= n * 2:
        recent = df["Volume"].tail(n).mean()
        prior = df["Volume"].tail(n * 2).head(n).mean()
        if prior and not pd.isna(prior):
            volume_ratio = recent / prior
            if volume_ratio >= 1.3:
                flow_score += 8
                reasons.append(f"최근 {n}일 거래량 증가")
            elif volume_ratio <= 0.8:
                flow_score -= 6
    if obv is not None and obv_ma20 is not None and not pd.isna(obv) and not pd.isna(obv_ma20):
        if obv > obv_ma20:
            flow_score += 10
            reasons.append("OBV 상승 우위")
        else:
            flow_score -= 10
    if current_price is not None:
        base_close = float(latest["Close"])
        gap_pct = (price - base_close) / base_close * 100
        if gap_pct >= 2:
            flow_score += 4
            reasons.append("연장장 강세")
        elif gap_pct <= -2:
            flow_score -= 4
    factor_scores["flow"] = _clamp_score(flow_score)

    relative_score, relative_reasons = _score_relative_strength(sector_comp)
    factor_scores["relative"] = relative_score
    reasons.extend(relative_reasons)

    total = _weighted_average(factor_scores, cfg["score_weights"])
    return total, reasons, {k: _clamp_score(v) for k, v in factor_scores.items()}


def score_fundamentals(
    fin: dict,
    df: pd.DataFrame,
    current_price: Optional[float] = None,
    sector_comp: Optional[dict] = None,
) -> tuple[int, list[str], FactorMap]:
    latest = df.iloc[-1]
    price = float(current_price) if current_price is not None else float(latest["Close"])
    high_52w = latest.get("HIGH_52W")
    low_52w = latest.get("LOW_52W")

    reasons = []
    factor_scores: dict[str, float] = {}

    ar = fin.get("annual_revenue_yoy_pct")
    qr = fin.get("quarterly_revenue_yoy_pct")
    ani = fin.get("annual_net_income_yoy_pct")
    qni = fin.get("quarterly_net_income_yoy_pct")
    fcf = fin.get("free_cash_flow")
    ocf = fin.get("operating_cash_flow")
    de = fin.get("debt_to_equity_pct")
    gross_margin = fin.get("gross_margin_pct")
    operating_margin = fin.get("operating_margin_pct")
    current_ratio = fin.get("current_ratio")
    share_change = fin.get("share_change_yoy_pct")
    earnings_quality = fin.get("earnings_quality_ratio")
    roe = fin.get("roe_pct")

    growth_score = 50.0
    for value, strong, positive, penalty in ((ar, 15, 5, -8), (qr, 10, 3, -6), (ani, 15, 7, -10), (qni, 10, 4, -7)):
        if value is None:
            continue
        if value >= strong:
            growth_score += 10
        elif value >= positive:
            growth_score += 5
        elif value < -10:
            growth_score += penalty
    if growth_score >= 60:
        reasons.append("매출/이익 성장 양호")
    elif growth_score <= 40:
        reasons.append("매출/이익 성장 둔화")
    factor_scores["growth"] = _clamp_score(growth_score)

    quality_score = 50.0
    if ocf is not None:
        quality_score += 8 if ocf > 0 else -10
    if fcf is not None:
        quality_score += 10 if fcf > 0 else -12
    if earnings_quality is not None:
        if earnings_quality >= 1.0:
            quality_score += 10
            reasons.append("현금흐름 기반 이익의 질 양호")
        elif earnings_quality < 0.7:
            quality_score -= 10
    factor_scores["quality"] = _clamp_score(quality_score)

    balance_score = 50.0
    if de is not None:
        if de < 50:
            balance_score += 15
            reasons.append("부채비율 매우 안정")
        elif de < 100:
            balance_score += 8
        elif de >= 200:
            balance_score -= 15
            reasons.append("부채 부담 높음")
    if current_ratio is not None:
        if current_ratio >= 1.5:
            balance_score += 8
        elif current_ratio < 1.0:
            balance_score -= 10
    if share_change is not None:
        if share_change <= 0:
            balance_score += 5
        elif share_change >= 3:
            balance_score -= 7
            reasons.append("주식수 희석 진행")
    factor_scores["balance"] = _clamp_score(balance_score)

    profitability_score = 50.0
    if gross_margin is not None:
        if gross_margin >= 40:
            profitability_score += 10
        elif gross_margin < 15:
            profitability_score -= 10
    if operating_margin is not None:
        if operating_margin >= 15:
            profitability_score += 12
            reasons.append("영업이익률 우수")
        elif operating_margin < 5:
            profitability_score -= 12
    if roe is not None:
        if roe >= 15:
            profitability_score += 10
            reasons.append("ROE 우수")
        elif roe < 5:
            profitability_score -= 8
    factor_scores["profitability"] = _clamp_score(profitability_score)

    market_position_score = 50.0
    if high_52w is not None and low_52w is not None and not pd.isna(high_52w) and not pd.isna(low_52w) and high_52w != low_52w:
        pos = (price - low_52w) / (high_52w - low_52w) * 100
        if 35 <= pos <= 80:
            market_position_score += 6
        elif pos > 95:
            market_position_score -= 8
            reasons.append("52주 고점 과열권 근접")
    relative_score, relative_reasons = _score_relative_strength(sector_comp)
    market_position_score = (market_position_score + relative_score) / 2
    reasons.extend([r for r in relative_reasons if "언더퍼폼" in r or "아웃퍼폼" in r])
    factor_scores["market_position"] = _clamp_score(market_position_score)

    total = _weighted_average(factor_scores, FUNDAMENTAL_SCORE_WEIGHTS)
    return total, reasons, {k: _clamp_score(v) for k, v in factor_scores.items()}


def combine_scores(tech_score: int, fund_score: int, term: str = "medium") -> tuple[int, str]:
    if term == "short":
        total = round(tech_score * 0.75 + fund_score * 0.25)
    elif term == "long":
        total = round(tech_score * 0.40 + fund_score * 0.60)
    else:
        total = round(tech_score * 0.55 + fund_score * 0.45)

    if total >= 80:
        label = "강한 매수"
    elif total >= 65:
        label = "매수 우세"
    elif total >= 45:
        label = "중립"
    elif total >= 30:
        label = "매도 우세"
    else:
        label = "강한 매도"

    return total, label
