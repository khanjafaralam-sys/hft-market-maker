import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
import os

# Configuration
PROCESSED_DIR = "data/processed"
SYMBOL = "BTCUSDT"

# Avellaneda-Stoikov & Alpha Parameters
RISK_AVERSION = 0.1       # How much we penalize holding inventory
SPREAD_HALF = 0.5         # How far away from the reservation price we place our quotes
ALPHA_SCALAR = 2.0        # How heavily we trust our ML prediction
MAX_INVENTORY = 10        # Maximum BTC exposure we will allow

def run_market_maker():
    print("Loading data & retraining Alpha Model...")
    file_pattern = f"{PROCESSED_DIR}/{SYMBOL}_features_*.parquet"
    df = pl.read_parquet(file_pattern).sort(["open_time"])
    
    # Feature Engineering (Same as before)
    df = df.with_columns([
        (pl.col("close").shift(-1) - pl.col("close")).alias("target_return"),
        pl.col("order_flow_imbalance").shift(1).alias("ofi_lag_1"),
        pl.col("order_flow_imbalance").shift(2).alias("ofi_lag_2"),
        pl.col("trade_count").alias("volatility_proxy")
    ]).drop_nulls()

    # Train / Test Split
    train_size = int(df.height * 0.8)
    train_df = df.slice(0, train_size)
    test_df = df.slice(train_size, df.height - train_size)
    
    features = ["order_flow_imbalance", "ofi_lag_1", "ofi_lag_2", "volatility_proxy"]
    
    # Train Model
    X_train = train_df.select(features).to_numpy()
    y_train = train_df.select("target_return").to_numpy().ravel()
    model = Ridge(alpha=1.0).fit(X_train, y_train)

    # Get Predictions for Test Set
    X_test = test_df.select(features).to_numpy()
    test_df = test_df.with_columns(
        pl.Series(name="alpha_prediction", values=model.predict(X_test))
    )

    print("\nStarting High-Frequency Backtest (Out-of-Sample)...")
    
    # Simulation Variables
    inventory = 0.0
    cash = 0.0
    pnl_history = []
    
    # Convert Polars DataFrame to dicts for fast iteration
    test_rows = test_df.to_dicts()

    for row in test_rows:
        mid_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        alpha = row["alpha_prediction"] * ALPHA_SCALAR
        
        # 1. Avellaneda-Stoikov Reservation Price
        reservation_price = mid_price - (inventory * RISK_AVERSION) + alpha
        
        # 2. Set Optimal Quotes
        our_bid = reservation_price - SPREAD_HALF
        our_ask = reservation_price + SPREAD_HALF
        
        # 3. Simulate Execution (Did the market cross our quotes?)
        # If the actual market low went lower than our bid, our bid got filled.
        if low_price < our_bid and inventory < MAX_INVENTORY:
            cash -= our_bid  # We spent cash
            inventory += 1.0 # We gained 1 BTC
            
        # If the actual market high went higher than our ask, our ask got filled.
        if high_price > our_ask and inventory > -MAX_INVENTORY:
            cash += our_ask  # We gained cash
            inventory -= 1.0 # We lost 1 BTC
            
        # 4. Mark to Market PnL
        current_pnl = cash + (inventory * mid_price)
        pnl_history.append(current_pnl)

    # 5. Output Results
    total_trades = len(pnl_history)
    final_pnl = pnl_history[-1]
    
    print("\n--- HFT EXECUTION REPORT ---")
    print(f"Simulation Steps: {total_trades}")
    print(f"Final Inventory: {inventory} BTC")
    print(f"Net Profit (Simulated): ${final_pnl:.2f}")

if __name__ == "__main__":
    run_market_maker()