"""Process local FNSPID raw data files for simulation."""

import argparse
import logging
import re
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_news_date(date_str):
    """Parse news dates like 'January 17, 2024 — 08:52 am EST' to datetime."""
    try:
        # Extract just the date part before the time
        date_part = date_str.split('—')[0].strip()
        return pd.to_datetime(date_part, format='%B %d, %Y')
    except:
        return pd.NaT


def process_fnspid_news(news_file: Path, ticker: str) -> pd.DataFrame:
    """Process FNSPID news data."""
    logger.info(f"Loading news from {news_file}...")
    
    # Load with encoding handling
    df = pd.read_csv(news_file, encoding='utf-8', encoding_errors='ignore')
    logger.info(f"Loaded {len(df)} news records")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Rename columns to standard format
    column_map = {
        'Date': 'date',
        'Text': 'body',
        'Url': 'url'
    }
    df = df.rename(columns=column_map)
    
    # Parse dates
    logger.info("Parsing dates...")
    df['date'] = df['date'].apply(parse_news_date)
    df = df.dropna(subset=['date'])
    
    # Create headline from first sentence of body
    logger.info("Creating headlines...")
    df['headline'] = df['body'].astype(str).apply(
        lambda x: x.split('.')[0][:100] if pd.notna(x) else 'Market update'
    )
    
    # Add ticker
    df['ticker'] = ticker.upper()
    
    # Add neutral sentiment (we'll enhance this later if needed)
    df['sentiment_hint'] = 'neutral'
    
    # Keep only needed columns
    df = df[['date', 'headline', 'body', 'ticker', 'sentiment_hint']]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'headline'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Processed {len(df)} news records")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def process_fnspid_prices(price_file: Path, ticker: str) -> pd.DataFrame:
    """Process FNSPID price data."""
    logger.info(f"Loading prices from {price_file}...")
    
    df = pd.read_csv(price_file, encoding='utf-8', encoding_errors='ignore')
    logger.info(f"Loaded {len(df)} price records")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Rename columns to standard format
    column_map = {
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    }
    df = df.rename(columns=column_map)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Add ticker
    df['ticker'] = ticker.upper()
    
    # Convert numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing prices
    df = df.dropna(subset=['close'])
    
    # Keep only needed columns
    cols_to_keep = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in cols_to_keep if col in df.columns]]
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Processed {len(df)} price records")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def filter_by_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter dataframe by date range."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return df[(df['date'] >= start) & (df['date'] <= end)]


def main():
    parser = argparse.ArgumentParser(description="Process local FNSPID data")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date")
    parser.add_argument("--news-file", type=Path, default=None, help="News raw file")
    parser.add_argument("--price-file", type=Path, default=None, help="Price raw file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Set defaults
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[1] / "data"
    
    if args.news_file is None:
        args.news_file = args.output_dir / "fnspid_raw" / "news_raw.csv"
    
    if args.price_file is None:
        args.price_file = args.output_dir / "fnspid_raw" / "prices_raw.csv"
    
    print("=" * 70)
    print(f"Processing FNSPID Data for {args.ticker}")
    print("=" * 70)
    
    # Process news
    news_df = process_fnspid_news(args.news_file, args.ticker)
    
    # Process prices
    price_df = process_fnspid_prices(args.price_file, args.ticker)
    
    # Filter by date range
    logger.info(f"Filtering data from {args.start} to {args.end}...")
    news_df = filter_by_date_range(news_df, args.start, args.end)
    price_df = filter_by_date_range(price_df, args.start, args.end)
    
    logger.info(f"After filtering: {len(news_df)} news, {len(price_df)} prices")
    
    if len(news_df) == 0:
        logger.error(f"No news found for {args.ticker} in date range")
        return 1
    
    if len(price_df) == 0:
        logger.error(f"No prices found for {args.ticker} in date range")
        return 1
    
    # Save processed files
    news_output = args.output_dir / f"fnspid_{args.ticker}_news.csv"
    price_output = args.output_dir / f"fnspid_{args.ticker}_prices.csv"
    
    news_df.to_csv(news_output, index=False)
    price_df.to_csv(price_output, index=False)
    
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"News: {len(news_df)} records")
    print(f"  Date range: {news_df['date'].min().date()} to {news_df['date'].max().date()}")
    print(f"  Saved to: {news_output}")
    print(f"\nPrices: {len(price_df)} records")
    print(f"  Date range: {price_df['date'].min().date()} to {price_df['date'].max().date()}")
    print(f"  Saved to: {price_output}")
    print("\n" + "=" * 70)
    print("Ready to run simulation!")
    print(f"Run: python scripts/run_fnspid_experiment.py --ticker {args.ticker}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

