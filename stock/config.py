TERM_CONFIG = {
    "short": {
        "ma_fast": 5,
        "ma_slow": 20,
        "volume_days": 5,
        "sr_days": 40,
        "rsi_period": 7,
        "rsi_neutral_low": 48,
        "rsi_neutral_high": 68,
        "atr_high_pct": 7.5,
        "score_weights": {
            "trend": 0.28,
            "momentum": 0.26,
            "volatility": 0.14,
            "flow": 0.18,
            "relative": 0.14,
        },
    },
    "medium": {
        "ma_fast": 20,
        "ma_slow": 60,
        "volume_days": 20,
        "sr_days": 90,
        "rsi_period": 14,
        "rsi_neutral_low": 45,
        "rsi_neutral_high": 65,
        "atr_high_pct": 6.0,
        "score_weights": {
            "trend": 0.30,
            "momentum": 0.22,
            "volatility": 0.12,
            "flow": 0.16,
            "relative": 0.20,
        },
    },
    "long": {
        "ma_fast": 60,
        "ma_slow": 120,
        "volume_days": 60,
        "sr_days": 180,
        "rsi_period": 21,
        "rsi_neutral_low": 40,
        "rsi_neutral_high": 60,
        "atr_high_pct": 5.0,
        "score_weights": {
            "trend": 0.26,
            "momentum": 0.14,
            "volatility": 0.10,
            "flow": 0.10,
            "relative": 0.15,
            "fundamental_overlay": 0.25,
        },
    },
}


RELATIVE_PERIOD_LABELS = (
    ("3m", "3개월"),
    ("6m", "6개월"),
    ("1y", "1년"),
)


FUNDAMENTAL_SCORE_WEIGHTS = {
    "growth": 0.30,
    "quality": 0.25,
    "balance": 0.20,
    "profitability": 0.20,
    "market_position": 0.05,
}
