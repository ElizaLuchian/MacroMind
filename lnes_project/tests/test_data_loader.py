import pandas as pd

from src.data_loader import _normalize_sentiment, _rename_by_priority, merge_news_and_prices


def test_merge_news_and_prices_supports_multi_column_join():
    news = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAA", "AAA"],
            "headline": ["h1", "h2"],
            "body": ["b1", "b2"],
            "sentiment_hint": ["positive", "negative"],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAA", "AAA"],
            "open": [10.0, 11.0],
            "close": [11.0, 10.5],
            "volume": [100, 120],
        }
    )

    merged = merge_news_and_prices(news, prices, join_on=("date", "ticker"))

    assert len(merged) == 2
    assert merged["close"].tolist() == [11.0, 10.5]
    assert (merged["ticker"] == "AAA").all()


def test_rename_by_priority_renames_first_available_column():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Article_title": ["Headline"]})
    renamed = _rename_by_priority(df, {"date": ("date", "Date"), "headline": ("headline", "Article_title")})
    assert "date" in renamed.columns
    assert "headline" in renamed.columns


def test_normalize_sentiment_maps_common_variants():
    series = pd.Series(["Pos", "NEG", "neutral", "unknown"])
    normalized = _normalize_sentiment(series)
    assert normalized.tolist() == ["positive", "negative", "neutral", "unknown"]


