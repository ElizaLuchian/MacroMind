"""Download real stock market data from Yahoo Finance and format it for LNES experiments."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_stock_data(ticker: str, start_date: str, end_date: str, output_dir: Path) -> None:
    """Download stock price data from Yahoo Finance."""
    print(f"[*] Downloading {ticker} data from {start_date} to {end_date}...")
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker} in the specified date range.")
    
    # Format to match expected structure
    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    # Keep only needed columns
    df = df[["date", "open", "high", "low", "close", "volume"]]
    
    # Format date as string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    # Save to CSV
    output_file = output_dir / f"{ticker}_prices.csv"
    df.to_csv(output_file, index=False)
    print(f"[+] Saved {len(df)} price records to: {output_file}")
    
    return df


def generate_synthetic_news(price_df: pd.DataFrame, ticker: str, output_dir: Path) -> None:
    """Generate synthetic news headlines based on price movements."""
    print(f"[*] Generating synthetic news headlines for {ticker}...")
    
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    
    news_data = []
    
    for idx, row in price_df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        close = row["close"]
        
        # Calculate price change if not first row
        if idx > 0:
            prev_close = price_df.iloc[idx - 1]["close"]
            pct_change = ((close - prev_close) / prev_close) * 100
        else:
            pct_change = 0
        
        # Generate headline and sentiment based on price movement
        if pct_change > 2:
            headline = f"{ticker} surges as investors show strong confidence"
            sentiment = "positive"
        elif pct_change > 0.5:
            headline = f"{ticker} gains on positive market sentiment"
            sentiment = "positive"
        elif pct_change < -2:
            headline = f"{ticker} drops amid market concerns"
            sentiment = "negative"
        elif pct_change < -0.5:
            headline = f"{ticker} declines as investors remain cautious"
            sentiment = "negative"
        else:
            headline = f"{ticker} trades sideways with mixed signals"
            sentiment = "neutral"
        
        body = f"{headline}. Trading volume: {int(row['volume']):,}. Analysts continue to monitor market conditions."
        
        news_data.append({
            "date": date,
            "headline": headline,
            "body": body,
            "sentiment_hint": sentiment,
            "ticker": ticker
        })
    
    news_df = pd.DataFrame(news_data)
    output_file = output_dir / f"{ticker}_news.csv"
    news_df.to_csv(output_file, index=False)
    print(f"[+] Saved {len(news_df)} news records to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download real stock data and generate synthetic news")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: data/)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[1] / "data"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Downloading Real Market Data for {args.ticker}")
    print("=" * 70)
    
    # Download price data
    price_df = download_stock_data(args.ticker, args.start, args.end, args.output_dir)
    
    # Generate synthetic news
    generate_synthetic_news(price_df, args.ticker, args.output_dir)
    
    print("\n" + "=" * 70)
    print("Done! You can now run experiments with:")
    print(f"   python scripts/run_experiment.py --dataset small --data-dir {args.output_dir}")
    print(f"   (Use {args.ticker}_news.csv and {args.ticker}_prices.csv)")
    print("=" * 70)


if __name__ == "__main__":
    main()

