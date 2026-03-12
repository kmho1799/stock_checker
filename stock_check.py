"""
Yahoo Finance 기반 주식 데이터 수집 및 투자 분석
- 단기/중기/장기: 추세(이동평균), 힘(거래량), 속도(RSI·MACD), 변동성(볼린저밴드), 가격대(지지·저항)
- 장기 보강: 고점/저점, 매출/이익 성장, 현금흐름, 부채비율
- 점수화: 기술적 + 장기 재무를 합쳐 매수/매도 신호 점수화
- 추가: 프리마켓/애프터마켓 가격 반영
"""

import argparse

from stock.runner import run_analysis


def main():
    parser = argparse.ArgumentParser(description="Yahoo Finance 주식 데이터 분석")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="종목 심볼 (예: AAPL, 005930.KS)")
    parser.add_argument(
        "-p",
        "--period",
        default="2y",
        choices=["3mo", "6mo", "1y", "2y", "5y"],
        help="조회 기간"
    )
    parser.add_argument(
        "-t",
        "--term",
        default="all",
        choices=["short", "medium", "long", "all"],
        help="분석 기간"
    )
    parser.add_argument("--no-recent", action="store_true", help="최근 5일 OHLC 미출력")
    args = parser.parse_args()

    run_analysis(
        args.ticker,
        period=args.period,
        show_chart_data=not args.no_recent,
        term=args.term,
    )


if __name__ == "__main__":
    main()
