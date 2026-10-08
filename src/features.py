"""
Module: Feature Engineering Pipeline
Generates time-series lag features, rolling statistics, calendar/event indicators,
and price relative features for demand forecasting models.
"""

import os
import sys
import time
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}%)")
    return df


def generate_features(df_long, config):
    logger.info("Starting feature engineering generation...")
    
    # Ensure sorting by series ID and day index
    df = df_long.sort_values(['id', 'd_num']).reset_index(drop=True)
    
    # 1. Temporal & Calendar Features
    logger.info("Creating calendar and date features...")
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_month'] = df['date'].dt.day.astype(np.int8)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(np.int8)
    df['is_weekend'] = df['wday'].isin([1, 2]).astype(np.int8)  # 1 and 2 in M5 calendar are Saturday & Sunday
    
    # Store-specific SNAP flag
    snap_conditions = [
        (df['state_id'] == 'CA') & (df['snap_CA'] == 1),
        (df['state_id'] == 'TX') & (df['snap_TX'] == 1),
        (df['state_id'] == 'WI') & (df['snap_WI'] == 1)
    ]
    df['snap_active'] = np.select(snap_conditions, [1, 1, 1], default=0).astype(np.int8)
    
    # Fill event NAs
    df['event_name_1'] = df['event_name_1'].fillna('None')
    df['event_type_1'] = df['event_type_1'].fillna('None')
    df['has_event'] = (df['event_name_1'] != 'None').astype(np.int8)
    
    # 2. Price Features
    logger.info("Creating price relative and volatility features...")
    item_price_mean = df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
    item_price_max = df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
    
    df['price_rel_mean'] = (df['sell_price'] / item_price_mean).astype(np.float32)
    df['price_discount_ratio'] = (df['sell_price'] / item_price_max).astype(np.float32)
    
    # 3. Demand Lag Features per Item-Store series
    logger.info("Computing demand lag features...")
    lags = config["features"].get("lags", [7, 14, 21, 28, 35, 42])
    grouped = df.groupby('id')['sales']
    
    for lag in lags:
        df[f'lag_{lag}'] = grouped.shift(lag).astype(np.float32)
        
    # 4. Rolling Window Demand Statistics (shift by forecast horizon=28 to prevent test data leakage)
    logger.info("Computing rolling window statistics...")
    windows = config["features"].get("rolling_windows", [7, 14, 28])
    
    # We use lag_28 as the base for rolling windows to match the 28-day forecasting horizon
    grouped_lag28 = df.groupby('id')['lag_28']
    
    for w in windows:
        df[f'rolling_mean_{w}'] = grouped_lag28.transform(lambda x: x.rolling(w, min_periods=1).mean()).astype(np.float32)
        df[f'rolling_std_{w}'] = grouped_lag28.transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0).astype(np.float32)
        df[f'rolling_max_{w}'] = grouped_lag28.transform(lambda x: x.rolling(w, min_periods=1).max()).astype(np.float32)
        df[f'rolling_min_{w}'] = grouped_lag28.transform(lambda x: x.rolling(w, min_periods=1).min()).astype(np.float32)

    # Convert categorical text columns to category dtype for LightGBM
    cat_cols = config["features"].get("categorical_cols", [])
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    df = reduce_mem_usage(df)
    return df


def run_feature_engineering(config_path="configs/config.yaml"):
    start_time = time.time()
    config = load_config(config_path)
    
    processed_dir = Path(config["paths"]["processed_dir"])
    input_parquet = processed_dir / "sales_long.parquet"
    output_parquet = processed_dir / "train_features.parquet"
    
    logger.info(f"Reading processed dataset from {input_parquet}...")
    df_long = pd.read_parquet(input_parquet)
    
    engineered_df = generate_features(df_long, config)
    
    logger.info(f"Saving engineered feature dataset ({engineered_df.shape[0]:,} rows, {engineered_df.shape[1]} columns) to {output_parquet}...")
    engineered_df.to_parquet(output_parquet, index=False)
    
    elapsed = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed:.2f} seconds.")
    return engineered_df


if __name__ == "__main__":
    run_feature_engineering()
