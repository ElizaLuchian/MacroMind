"""Download and clean the FNSPID dataset, handling encoding issues."""

import argparse
import logging
from pathlib import Path
import requests
import pandas as pd
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_and_clean_fnspid_news(output_dir: Path, limit: int = None) -> pd.DataFrame:
    """Download FNSPID news data and clean encoding issues."""
    logger.info("Downloading FNSPID news dataset...")
    
    # FNSPID dataset URLs from Hugging Face repository
    news_urls = [
        "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv",
        "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
    ]
    
    all_news = []
    
    for idx, url in enumerate(news_urls):
        logger.info(f"Downloading news file {idx + 1}/{len(news_urls)}...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Try different encodings to handle the data
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(
                        BytesIO(response.content),
                        encoding=encoding,
                        on_bad_lines='skip',
                        low_memory=False
                    )
                    logger.info(f"Successfully loaded with {encoding} encoding: {len(df)} rows")
                    all_news.append(df)
                    break
                except Exception as e:
                    if encoding == 'cp1252':
                        logger.error(f"Failed to load file {idx + 1}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            continue
    
    if not all_news:
        raise ValueError("Failed to download any news data")
    
    # Combine all news dataframes
    logger.info("Combining news data...")
    news_df = pd.concat(all_news, ignore_index=True)
    logger.info(f"Total news records: {len(news_df)}")
    
    # Clean and standardize columns
    logger.info("Cleaning news data...")
    
    # Map column names to standard format
    column_mapping = {
        'Date': 'date',
        'date': 'date',
        'publish_date': 'date',
        'Headline': 'headline',
        'headline': 'headline',
        'title': 'headline',
        'Title': 'headline',
        'Article_title': 'headline',
        'Body': 'body',
        'body': 'body',
        'content': 'body',
        'Article': 'body',
        'Ticker': 'ticker',
        'ticker': 'ticker',
        'symbol': 'ticker',
        'Symbol': 'ticker',
        'Stock': 'ticker',
        'Sentiment': 'sentiment_hint',
        'sentiment': 'sentiment_hint',
        'sentiment_label': 'sentiment_hint',
        'Finbert_sentiment': 'sentiment_hint'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in news_df.columns:
            news_df = news_df.rename(columns={old_name: new_name})
    
    # Ensure required columns exist
    required_cols = ['date', 'headline', 'ticker']
    missing = [col for col in required_cols if col not in news_df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.info(f"Available columns: {list(news_df.columns)}")
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add body column if missing
    if 'body' not in news_df.columns:
        news_df['body'] = news_df['headline']
    
    # Add sentiment if missing
    if 'sentiment_hint' not in news_df.columns:
        news_df['sentiment_hint'] = 'neutral'
    
    # Clean data
    news_df = news_df.dropna(subset=['date', 'headline', 'ticker'])
    
    # Parse dates
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df = news_df.dropna(subset=['date'])
    
    # Clean text fields - remove or replace problematic characters
    for col in ['headline', 'body']:
        if col in news_df.columns:
            news_df[col] = news_df[col].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')
    
    # Normalize sentiment
    if 'sentiment_hint' in news_df.columns:
        sentiment_map = {
            'pos': 'positive', 'positive': 'positive', '1': 'positive', 'bullish': 'positive',
            'neg': 'negative', 'negative': 'negative', '-1': 'negative', 'bearish': 'negative',
            'neu': 'neutral', 'neutral': 'neutral', '0': 'neutral'
        }
        news_df['sentiment_hint'] = news_df['sentiment_hint'].astype(str).str.lower().str.strip()
        news_df['sentiment_hint'] = news_df['sentiment_hint'].map(lambda x: sentiment_map.get(x, 'neutral'))
    
    # Sort by date
    news_df = news_df.sort_values('date').reset_index(drop=True)
    
    # Apply limit if specified
    if limit:
        news_df = news_df.head(limit)
    
    # Save cleaned data
    output_file = output_dir / "fnspid_news_cleaned.csv"
    news_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved {len(news_df)} cleaned news records to: {output_file}")
    
    return news_df


def download_and_clean_fnspid_prices(output_dir: Path, limit: int = None) -> pd.DataFrame:
    """Download FNSPID price data and clean encoding issues."""
    logger.info("Downloading FNSPID price dataset...")
    
    url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.csv"
    
    try:
        logger.info("Downloading price data...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                price_df = pd.read_csv(
                    BytesIO(response.content),
                    encoding=encoding,
                    on_bad_lines='skip',
                    low_memory=False
                )
                logger.info(f"Successfully loaded with {encoding} encoding: {len(price_df)} rows")
                break
            except Exception as e:
                if encoding == 'cp1252':
                    raise ValueError(f"Failed to load price data with any encoding: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to download price data: {e}")
        raise
    
    logger.info("Cleaning price data...")
    
    # Map column names
    column_mapping = {
        'Date': 'date',
        'date': 'date',
        'Ticker': 'ticker',
        'ticker': 'ticker',
        'symbol': 'ticker',
        'Symbol': 'ticker',
        'Open': 'open',
        'open': 'open',
        'High': 'high',
        'high': 'high',
        'Low': 'low',
        'low': 'low',
        'Close': 'close',
        'close': 'close',
        'AdjClose': 'close',
        'Adj Close': 'close',
        'Volume': 'volume',
        'volume': 'volume'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in price_df.columns:
            price_df = price_df.rename(columns={old_name: new_name})
    
    # Ensure required columns
    required_cols = ['date', 'ticker', 'close']
    missing = [col for col in required_cols if col not in price_df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.info(f"Available columns: {list(price_df.columns)}")
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean data
    price_df = price_df.dropna(subset=['date', 'ticker', 'close'])
    
    # Parse dates
    price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
    price_df = price_df.dropna(subset=['date'])
    
    # Convert numeric columns, coercing errors
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
    
    # Remove rows with non-numeric prices
    price_df = price_df.dropna(subset=['close'])
    
    # Sort by ticker and date
    price_df = price_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Apply limit if specified
    if limit:
        price_df = price_df.head(limit)
    
    # Save cleaned data
    output_file = output_dir / "fnspid_prices_cleaned.csv"
    price_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved {len(price_df)} cleaned price records to: {output_file}")
    
    return price_df


def main():
    parser = argparse.ArgumentParser(description="Download and clean FNSPID dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--news-limit", type=int, default=None, help="Limit news records")
    parser.add_argument("--price-limit", type=int, default=None, help="Limit price records")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[1] / "data"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Downloading and Cleaning FNSPID Dataset")
    print("=" * 70)
    
    try:
        # Download and clean news
        news_df = download_and_clean_fnspid_news(args.output_dir, args.news_limit)
        
        # Download and clean prices
        price_df = download_and_clean_fnspid_prices(args.output_dir, args.price_limit)
        
        # Show summary
        print("\n" + "=" * 70)
        print("Dataset Summary:")
        print("=" * 70)
        print(f"News records: {len(news_df)}")
        print(f"Price records: {len(price_df)}")
        print(f"\nNews date range: {news_df['date'].min()} to {news_df['date'].max()}")
        print(f"Price date range: {price_df['date'].min()} to {price_df['date'].max()}")
        print(f"\nTickers in news: {news_df['ticker'].nunique()}")
        print(f"Tickers in prices: {price_df['ticker'].nunique()}")
        
        # Show top tickers
        print("\nTop 10 tickers by news volume:")
        print(news_df['ticker'].value_counts().head(10).to_string())
        
        print("\n" + "=" * 70)
        print("Success! Cleaned data saved to:")
        print(f"  - {args.output_dir / 'fnspid_news_cleaned.csv'}")
        print(f"  - {args.output_dir / 'fnspid_prices_cleaned.csv'}")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to download/clean FNSPID dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

