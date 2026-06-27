import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
import itertools
import os

PROCESSED_DIR = "data/processed"
SYMBOL = "BTCUSDT"
MAX_INVENTORY = 10

def optimize_market_maker():
    print("Loading data & training Alpha Model for optimization...")
    file_pattern = f"{PROCESSED_DIR}/{SYMBOL}_features_*.parquet"
    df = pl.read_parquet(file_pattern).sort(["open_time"])
    
    # Feature Engineering
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
    
    X_train = train_df.select(features).to_numpy()
    y_train = train_df.select("target_return").to_numpy().ravel()
    model = Ridge(alpha=1.0).fit(X_train, y_train)

    X_test = test_df.select(features).to_numpy()
    test_df = test_df.with_columns(
        pl.Series(name="raw_alpha", values=model.predict(X_test))
    )
    test_rows = test_df.to_dicts()

    # Define Parameter Grid
    gamma_grid = [0.1, 0.5, 1.0, 2.5]       # Risk Aversion
    spread_grid = [0.5, 1.0, 2.0, 4.0]      # Quote Distance
    alpha_scalar_grid = [0.5, 2.0, 5.0, 10.0] # Signal Strength

    best_pnl = -float('inf')
    best_params = None
    
    print("\nRunning Parameter Grid Search...")
    
    for gamma, spread, alpha_scalar in itertools.product(gamma_grid, spread_grid, alpha_scalar_grid):
        inventory = 0.0
        cash = 0.0
        
        for row in test_rows:
            mid_price = row["open"]
            alpha = row["raw_alpha"] * alpha_scalar
            
            # AS Reservation Price
            reservation_price = mid_price - (inventory * gamma) + alpha
            
            our_bid = reservation_price - spread
            our_ask = reservation_price + spread
            
            # Execution
            if row["low"] < our_bid and inventory < MAX_INVENTORY:
                cash -= our_bid 
                inventory += 1.0 
                
            if row["high"] > our_ask and inventory > -MAX_INVENTORY:
                cash += our_ask 
                inventory -= 1.0 
                
        final_pnl = cash + (inventory * test_rows[-1]["close"])
        
        # Log if it's the best so far
        if final_pnl > best_pnl:
            best_pnl = final_pnl
            best_params = (gamma, spread, alpha_scalar)

    print("\n--- OPTIMIZATION RESULTS ---")
    print(f"Best Net Profit: ${best_pnl:.2f}")
    print(f"Optimal Risk Aversion (Gamma): {best_params[0]}")
    print(f"Optimal Spread Half: {best_params[1]}")
    print(f"Optimal Alpha Scalar: {best_params[2]}")

if __name__ == "__main__":
    optimize_market_maker()