"""Load FNSPID dataset directly from CSV with robust error handling."""

import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_fnspid_from_csv(ticker: str, start_date: str, end_date: str, output_dir: Path):
    """Load FNSPID data directly from HuggingFace CSV files."""
    
    logger.info(f"Loading FNSPID data for {ticker}")
    
    # Direct URLs to the CSV files on Hugging Face
    news_url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
    price_url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.csv"
    
    # Load news with error handling
    logger.info("Loading news data from CSV...")
    try:
        news_df = pd.read_csv(
            news_url,
            on_bad_lines='skip',  # Skip problematic lines
            encoding='utf-8',
            encoding_errors='ignore',  # Ignore encoding errors
            low_memory=False
        )
        logger.info(f"Loaded {len(news_df)} news records")
        logger.info(f"News columns: {list(news_df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load news: {e}")
        raise
    
    # Load prices with error handling  
    logger.info("Loading price data from CSV...")
    try:
        # This might be a zip file, try direct CSV first
        try:
            price_df = pd.read_csv(
                price_url,
                on_bad_lines='skip',
                encoding='utf-8',
                encoding_errors='ignore',
                low_memory=False
            )
        except:
            # Try alternative price source
            price_url_alt = "https://huggingface.co/datasets/Zihan1004/FNSPID/raw/main/Stock_price/full_history.csv"
            price_df = pd.read_csv(
                price_url_alt,
                on_bad_lines='skip',
                encoding='utf-8',
                encoding_errors='ignore',
                low_memory=False
            )
        logger.info(f"Loaded {len(price_df)} price records")
        logger.info(f"Price columns: {list(price_df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load prices: {e}")
        raise
    
    # Clean and standardize column names
    logger.info("Standardizing column names...")
    
    # News column mapping
    news_col_map = {}
    for col in news_df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['date', 'publish_date', 'publishdate']:
            news_col_map[col] = 'date'
        elif col_lower in ['headline', 'title', 'article_title']:
            news_col_map[col] = 'headline'
        elif col_lower in ['body', 'content', 'article', 'text']:
            news_col_map[col] = 'body'
        elif col_lower in ['ticker', 'symbol', 'stock_symbol', 'stock']:
            news_col_map[col] = 'ticker'
        elif 'sentiment' in col_lower:
            news_col_map[col] = 'sentiment_hint'
    
    news_df = news_df.rename(columns=news_col_map)
    
    # Price column mapping
    price_col_map = {}
    for col in price_df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'date':
            price_col_map[col] = 'date'
        elif col_lower in ['ticker', 'symbol', 'stock_symbol']:
            price_col_map[col] = 'ticker'
        elif col_lower == 'open':
            price_col_map[col] = 'open'
        elif col_lower == 'high':
            price_col_map[col] = 'high'
        elif col_lower == 'low':
            price_col_map[col] = 'low'
        elif col_lower in ['close', 'adjclose', 'adj_close']:
            price_col_map[col] = 'close'
        elif col_lower == 'volume':
            price_col_map[col] = 'volume'
    
    price_df = price_df.rename(columns=price_col_map)
    
    logger.info(f"News columns after mapping: {list(news_df.columns)}")
    logger.info(f"Price columns after mapping: {list(price_df.columns)}")
    
    # Validate required columns
    if 'date' not in news_df.columns or 'headline' not in news_df.columns or 'ticker' not in news_df.columns:
        raise ValueError(f"Missing required news columns. Have: {list(news_df.columns)}")
    
    if 'date' not in price_df.columns or 'ticker' not in price_df.columns or 'close' not in price_df.columns:
        raise ValueError(f"Missing required price columns. Have: {list(price_df.columns)}")
    
    # Add missing optional columns
    if 'body' not in news_df.columns:
        news_df['body'] = news_df['headline']
    
    if 'sentiment_hint' not in news_df.columns:
        news_df['sentiment_hint'] = 'neutral'
    
    # Parse dates
    logger.info("Parsing dates...")
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
    
    # Remove rows with invalid dates
    news_df = news_df.dropna(subset=['date'])
    price_df = price_df.dropna(subset=['date'])
    
    # Filter by ticker
    logger.info(f"Filtering for ticker {ticker}...")
    ticker_upper = ticker.upper()
    news_df = news_df[news_df['ticker'].astype(str).str.upper().str.strip() == ticker_upper]
    price_df = price_df[price_df['ticker'].astype(str).str.upper().str.strip() == ticker_upper]
    
    logger.info(f"After ticker filter: {len(news_df)} news, {len(price_df)} prices")
    
    # Filter by date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    news_df = news_df[(news_df['date'] >= start) & (news_df['date'] <= end)]
    price_df = price_df[(price_df['date'] >= start) & (price_df['date'] <= end)]
    
    logger.info(f"After date filter: {len(news_df)} news, {len(price_df)} prices")
    
    if len(news_df) == 0:
        raise ValueError(f"No news found for {ticker} between {start_date} and {end_date}")
    
    if len(price_df) == 0:
        raise ValueError(f"No prices found for {ticker} between {start_date} and {end_date}")
    
    # Clean price numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
    
    # Remove rows with missing critical data
    news_df = news_df.dropna(subset=['date', 'headline', 'ticker'])
    price_df = price_df.dropna(subset=['date', 'close', 'ticker'])
    
    # Sort by date
    news_df = news_df.sort_values('date').reset_index(drop=True)
    price_df = price_df.sort_values('date').reset_index(drop=True)
    
    # Save to CSV
    news_file = output_dir / f"fnspid_{ticker}_news.csv"
    price_file = output_dir / f"fnspid_{ticker}_prices.csv"
    
    news_df.to_csv(news_file, index=False)
    price_df.to_csv(price_file, index=False)
    
    logger.info(f"Saved: {news_file} ({len(news_df)} records)")
    logger.info(f"Saved: {price_file} ({len(price_df)} records)")
    
    return news_df, price_df


def main():
    parser = argparse.ArgumentParser(description="Load FNSPID data directly from CSV")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path, default=None)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[1] / "data"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Loading FNSPID Data for {args.ticker}")
    print("=" * 70)
    
    try:
        news_df, price_df = load_fnspid_from_csv(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir
        )
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"News: {len(news_df)} records")
        print(f"  Date range: {news_df['date'].min().date()} to {news_df['date'].max().date()}")
        print(f"Prices: {len(price_df)} records")
        print(f"  Date range: {price_df['date'].min().date()} to {price_df['date'].max().date()}")
        print("\nReady to run experiment!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

