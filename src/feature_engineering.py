import polars as pl
import os
from datetime import datetime

# Configuration
SYMBOL = "BTCUSDT"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
VOLUME_BUCKET_SIZE = 10.0  # Create a new "tick" every time 10 BTC is traded

def process_tick_data(date: str):
    print(f"Processing raw ticks for {date}...")
    trades_path = f"{RAW_DIR}/{SYMBOL}_trades_{date}.parquet"
    output_path = f"{PROCESSED_DIR}/{SYMBOL}_features_{date}.parquet"

    if not os.path.exists(trades_path):
        print(f"No trade data found for {date}")
        return

    # Load the raw trades
    df = pl.read_parquet(trades_path)

    # 1. Microstructure Engineering: Trade Sign
    # Binance 'is_buyer_maker' == True means a market SELL order hit the limit BUY book.
    # We map Market Buys to +1, Market Sells to -1
    df = df.with_columns(
        pl.when(pl.col("is_buyer_maker") == True)
        .then(-1)
        .otherwise(1)
        .alias("trade_sign")
    )

    # Calculate signed volume (positive for buy volume, negative for sell volume)
    df = df.with_columns(
        (pl.col("qty") * pl.col("trade_sign")).alias("signed_volume")
    )

    # 2. Volume Clock Generation (Advanced Quant Concept)
    # We calculate the cumulative volume traded, and create a "bucket ID" for every 10 BTC
    df = df.with_columns(
        pl.col("qty").cum_sum().alias("cum_volume")
    )
    df = df.with_columns(
        (pl.col("cum_volume") // VOLUME_BUCKET_SIZE).cast(pl.Int64).alias("volume_bucket")
    )

    # 3. Aggregating into Volume Bars
    print("Aggregating into Volume Clocks...")
    volume_bars = df.group_by("volume_bucket").agg([
        pl.col("timestamp").first().alias("open_time"),
        pl.col("timestamp").last().alias("close_time"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("qty").sum().alias("total_volume"),
        
        # Order Flow Imbalance (OFI) - Our primary Alpha Signal
        pl.col("signed_volume").sum().alias("order_flow_imbalance"),
        
        # Trade Count (Volatility proxy)
        pl.len().alias("trade_count")
    ]).sort("volume_bucket")

    # Save the processed features
    volume_bars.write_parquet(output_path, compression="snappy")
    print(f"Successfully engineered features: {output_path}")
    print(f"Compressed {len(df)} raw ticks into {len(volume_bars)} volume-normalized bars.\n")

if __name__ == "__main__":
    # Process the days we successfully downloaded
    dates_to_process = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]
    for d in dates_to_process:
        process_tick_data(d)