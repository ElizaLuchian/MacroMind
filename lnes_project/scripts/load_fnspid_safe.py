"""Load FNSPID dataset with safe error handling for encoding issues."""

import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_fnspid_safe(
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    max_news: int = 5000,
    max_prices: int = 10000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FNSPID using datasets library with error recovery."""
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    logger.info(f"Loading FNSPID data for {ticker} from {start_date} to {end_date}")
    
    # Load with streaming to handle errors better
    logger.info("Loading news data (this may take a moment)...")
    
    try:
        # Load without split slicing first, then filter in pandas
        news_dataset = load_dataset(
            "Zihan1004/FNSPID",
            split="news",
            streaming=True,  # Use streaming to avoid loading all at once
        )
        
        # Convert to pandas iteratively with error handling
        news_records = []
        count = 0
        
        for record in news_dataset:
            if count >= max_news:
                break
            
            try:
                # Clean record to avoid encoding issues
                clean_record = {}
                for key, value in record.items():
                    if isinstance(value, str):
                        # Remove non-ASCII characters
                        clean_record[key] = value.encode('ascii', 'ignore').decode('ascii')
                    else:
                        clean_record[key] = value
                
                news_records.append(clean_record)
                count += 1
                
                if count % 1000 == 0:
                    logger.info(f"Processed {count} news records...")
                    
            except Exception as e:
                # Skip problematic records
                continue
        
        news_df = pd.DataFrame(news_records)
        logger.info(f"Loaded {len(news_df)} news records")
        
    except Exception as e:
        logger.error(f"Error loading news: {e}")
        raise
    
    # Load price data with streaming
    logger.info("Loading price data...")
    
    try:
        price_dataset = load_dataset(
            "Zihan1004/FNSPID",
            split="price",
            streaming=True,
        )
        
        price_records = []
        count = 0
        
        for record in price_dataset:
            if count >= max_prices:
                break
            
            try:
                # Clean record
                clean_record = {}
                for key, value in record.items():
                    if isinstance(value, str):
                        clean_record[key] = value.encode('ascii', 'ignore').decode('ascii')
                    else:
                        clean_record[key] = value
                
                price_records.append(clean_record)
                count += 1
                
                if count % 5000 == 0:
                    logger.info(f"Processed {count} price records...")
                    
            except Exception as e:
                continue
        
        price_df = pd.DataFrame(price_records)
        logger.info(f"Loaded {len(price_df)} price records")
        
    except Exception as e:
        logger.error(f"Error loading prices: {e}")
        raise
    
    # Now clean and filter the data
    logger.info("Cleaning and filtering data...")
    
    # Standardize column names for news
    news_col_map = {
        'Date': 'date', 'publish_date': 'date',
        'Headline': 'headline', 'title': 'headline', 'Title': 'headline',
        'Body': 'body', 'content': 'body',
        'Ticker': 'ticker', 'symbol': 'ticker', 'Symbol': 'ticker',
        'Sentiment': 'sentiment_hint', 'sentiment_label': 'sentiment_hint'
    }
    
    for old, new in news_col_map.items():
        if old in news_df.columns and new not in news_df.columns:
            news_df = news_df.rename(columns={old: new})
    
    # Standardize column names for prices
    price_col_map = {
        'Date': 'date',
        'Ticker': 'ticker', 'symbol': 'ticker', 'Symbol': 'ticker',
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'AdjClose': 'close',
        'Volume': 'volume'
    }
    
    for old, new in price_col_map.items():
        if old in price_df.columns and new not in price_df.columns:
            price_df = price_df.rename(columns={old: new})
    
    # Parse dates
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
    
    # Filter by ticker (case-insensitive)
    ticker_upper = ticker.upper()
    news_df = news_df[news_df['ticker'].str.upper() == ticker_upper]
    price_df = price_df[price_df['ticker'].str.upper() == ticker_upper]
    
    logger.info(f"After ticker filter: {len(news_df)} news, {len(price_df)} prices")
    
    # Filter by date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    news_df = news_df[(news_df['date'] >= start) & (news_df['date'] <= end)]
    price_df = price_df[(price_df['date'] >= start) & (price_df['date'] <= end)]
    
    logger.info(f"After date filter: {len(news_df)} news, {len(price_df)} prices")
    
    if len(news_df) == 0:
        raise ValueError(f"No news data found for {ticker} in date range {start_date} to {end_date}")
    
    if len(price_df) == 0:
        raise ValueError(f"No price data found for {ticker} in date range {start_date} to {end_date}")
    
    # Add missing columns
    if 'body' not in news_df.columns:
        news_df['body'] = news_df['headline']
    
    if 'sentiment_hint' not in news_df.columns:
        news_df['sentiment_hint'] = 'neutral'
    
    # Clean numeric columns in prices
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
    
    # Remove NaN values
    news_df = news_df.dropna(subset=['date', 'headline'])
    price_df = price_df.dropna(subset=['date', 'close'])
    
    # Sort by date
    news_df = news_df.sort_values('date').reset_index(drop=True)
    price_df = price_df.sort_values('date').reset_index(drop=True)
    
    # Save cleaned data
    news_file = output_dir / f"fnspid_{ticker}_news.csv"
    price_file = output_dir / f"fnspid_{ticker}_prices.csv"
    
    news_df.to_csv(news_file, index=False)
    price_df.to_csv(price_file, index=False)
    
    logger.info(f"Saved cleaned data:")
    logger.info(f"  News: {news_file} ({len(news_df)} records)")
    logger.info(f"  Prices: {price_file} ({len(price_df)} records)")
    
    return news_df, price_df


def main():
    parser = argparse.ArgumentParser(description="Load FNSPID dataset with error handling")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker (e.g., AAPL)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--max-news", type=int, default=5000, help="Max news records to load")
    parser.add_argument("--max-prices", type=int, default=10000, help="Max price records to load")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[1] / "data"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Loading FNSPID Data for {args.ticker}")
    print("=" * 70)
    
    try:
        news_df, price_df = load_fnspid_safe(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir,
            max_news=args.max_news,
            max_prices=args.max_prices
        )
        
        print("\n" + "=" * 70)
        print("Success!")
        print("=" * 70)
        print(f"News: {len(news_df)} records from {news_df['date'].min()} to {news_df['date'].max()}")
        print(f"Prices: {len(price_df)} records from {price_df['date'].min()} to {price_df['date'].max()}")
        print("\nYou can now run the experiment with this data!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

