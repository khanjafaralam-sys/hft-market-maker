import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import os

# Configuration
PROCESSED_DIR = "data/processed"
SYMBOL = "BTCUSDT"

def train_alpha_model():
    print("Loading volume-clocked feature data...")
    
    # Load all processed parquet files at once
    # Using a wildcard to grab all 5 days
    file_pattern = f"{PROCESSED_DIR}/{SYMBOL}_features_*.parquet"
    
    try:
        df = pl.read_parquet(file_pattern).sort(["open_time"])
    except Exception as e:
        print(f"Error loading files. Ensure they exist in {PROCESSED_DIR}. Error: {e}")
        return

    # 1. Feature Engineering: Lags and Targets
    print("Generating predictive targets and lagged features...")
    
    # Target: The price return in the NEXT volume bar
    df = df.with_columns([
        (pl.col("close").shift(-1) - pl.col("close")).alias("target_return")
    ])
    
    # Features: The current and previous Order Flow Imbalances
    df = df.with_columns([
        pl.col("order_flow_imbalance").shift(1).alias("ofi_lag_1"),
        pl.col("order_flow_imbalance").shift(2).alias("ofi_lag_2"),
        pl.col("trade_count").alias("volatility_proxy")
    ])
    
    # Drop rows with nulls caused by shifting
    df = df.drop_nulls()

    # 2. Train / Test Split
    # We never randomly shuffle financial data. We train on the past to predict the future.
    # Let's use the first 80% of the bars for training, and the last 20% for testing.
    total_rows = df.height
    train_size = int(total_rows * 0.8)
    
    train_df = df.slice(0, train_size)
    test_df = df.slice(train_size, total_rows - train_size)
    
    features = ["order_flow_imbalance", "ofi_lag_1", "ofi_lag_2", "volatility_proxy"]
    
    X_train = train_df.select(features).to_numpy()
    y_train = train_df.select("target_return").to_numpy().ravel()
    
    X_test = test_df.select(features).to_numpy()
    y_test = test_df.select("target_return").to_numpy().ravel()

    # 3. Model Training
    print(f"Training ML Model on {train_size} bars...")
    # Ridge regression handles multicollinearity better than standard linear regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    print("Evaluating Model on unseen test data...")
    predictions = model.predict(X_test)
    
    # We care most about Directional Accuracy (Hit Rate). 
    # Did we correctly predict if the price would go up or down?
    actual_direction = np.sign(y_test)
    predicted_direction = np.sign(predictions)
    
    # Ignore flat bars (where price didn't move) for directional accuracy
    valid_indices = actual_direction != 0
    hit_rate = accuracy_score(actual_direction[valid_indices], predicted_direction[valid_indices])
    
    print("\n--- QUANTITATIVE ALPHA REPORT ---")
    print(f"Total Bars Evaluated: {total_rows}")
    print(f"Directional Hit Rate (Out of Sample): {hit_rate * 100:.2f}%")
    
    print("\nFeature Weights (How much each factor drives the price):")
    for feature, coef in zip(features, model.coef_):
        print(f"- {feature}: {coef:.5f}")

if __name__ == "__main__":
    train_alpha_model()