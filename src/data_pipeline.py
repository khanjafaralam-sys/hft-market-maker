import asyncio
import aiohttp
import zipfile
import io
import os
import polars as pl
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = "BTCUSDT"
MARKET_TYPE = "spot" # Pivoted to Spot for reliable Level 1 data
DATA_TYPES = ["trades", "bookTicker"]
START_DATE = "2026-06-01" 
END_DATE = "2026-06-05" 
BASE_URL = "https://data.binance.vision/data"
SAVE_DIR = "data/raw"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# ASYNC DOWNLOADER ENGINE
# ==========================================
async def download_and_convert(session: aiohttp.ClientSession, data_type: str, date: str):
    """Downloads a daily zip file from Binance, unzips it, and saves it as a Parquet file."""
    
    # URL routes to: https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2026-06-01.zip
    url = f"{BASE_URL}/{MARKET_TYPE}/monthly/{data_type}/{SYMBOL}/{SYMBOL}-{data_type}-{date}.zip"
    
    # Actually, Binance Vision uses "daily" for daily data. Let's ensure the route is perfect:
    url = f"{BASE_URL}/{MARKET_TYPE}/daily/{data_type}/{SYMBOL}/{SYMBOL}-{data_type}-{date}.zip"
    
    parquet_path = f"{SAVE_DIR}/{SYMBOL}_{data_type}_{date}.parquet"
    if os.path.exists(parquet_path):
        print(f"Skipping {data_type} for {date} (Already exists)")
        return

    print(f"Downloading {data_type} for {date}...")
    async with session.get(url) as response:
        if response.status == 200:
            # Read zip file straight into memory to save disk I/O
            zip_content = await response.read()
            with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                # The zip usually contains one CSV file named identically to the zip
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as csv_file:
                    
                    # Read into Polars and strictly define schemas to prevent ComputeErrors
                    if data_type == "bookTicker":
                        schema = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time", "event_time"]
                        # Spot market bookTicker has no event_time column, just transaction_time. Binance is sneaky like that.
                        schema = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "timestamp"]
                        df = pl.read_csv(csv_file.read(), has_header=False, new_columns=schema)
                    else: # trades
                        schema = ["id", "price", "qty", "quote_qty", "timestamp", "is_buyer_maker", "is_best_match"]
                        df = pl.read_csv(csv_file.read(), has_header=False, new_columns=schema)
                    
                    # Compress and save
                    df.write_parquet(parquet_path, compression="snappy")
                    print(f"Successfully saved {parquet_path}")
        else:
            print(f"Failed to download {url}. Status: {response.status}")

async def main():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    # Generate list of dates
    date_list = [(start + timedelta(days=x)).strftime("%Y-%m-%d") for x in range((end-start).days + 1)]
    
    # Limit connections to avoid overwhelming the server
    connector = aiohttp.TCPConnector(limit=10) 
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for date in date_list:
            for data_type in DATA_TYPES:
                tasks.append(download_and_convert(session, data_type, date))
        
        # Run all downloads concurrently
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())