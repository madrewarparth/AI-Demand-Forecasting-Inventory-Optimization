"""
Module: Data Ingestion & Reshaping Pipeline
Reshapes raw Walmart M5 sales dataset from wide format to long time-series format,
joins calendar and price tables, performs data quality checks, and outputs processed parquet files.
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
    """Iterate through dataframe columns to downcast numeric types and reduce memory footprint."""
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def perform_data_quality_checks(df_long):
    """Executes data quality checks on the reshaped dataset."""
    logger.info("--- Data Quality Checks ---")
    
    total_rows = len(df_long)
    missing_prices = df_long['sell_price'].isnull().sum()
    zero_sales = (df_long['sales'] == 0).sum()
    outlier_sales = (df_long['sales'] > (df_long['sales'].mean() + 5 * df_long['sales'].std())).sum()

    quality_summary = {
        "total_records": int(total_rows),
        "missing_prices_count": int(missing_prices),
        "missing_prices_pct": float(round((missing_prices / total_rows) * 100, 2)),
        "zero_sales_count": int(zero_sales),
        "zero_sales_pct": float(round((zero_sales / total_rows) * 100, 2)),
        "extreme_outliers_count": int(outlier_sales),
        "unique_items": int(df_long['item_id'].nunique()),
        "unique_stores": int(df_long['store_id'].nunique()),
        "date_range": [str(df_long['date'].min()), str(df_long['date'].max())]
    }

    logger.info(f"Total records processed: {total_rows:,}")
    logger.info(f"Missing prices: {missing_prices:,} ({quality_summary['missing_prices_pct']}%)")
    logger.info(f"Zero daily sales: {zero_sales:,} ({quality_summary['zero_sales_pct']}%)")
    logger.info(f"Extreme outliers (>5 std): {outlier_sales:,}")
    
    return quality_summary


def run_ingestion(config_path="configs/config.yaml"):
    start_time = time.time()
    config = load_config(config_path)
    
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    sales_path = raw_dir / config["data"]["raw_sales_file"]
    calendar_path = raw_dir / config["data"]["raw_calendar_file"]
    prices_path = raw_dir / config["data"]["raw_prices_file"]
    
    logger.info(f"Loading raw dataset files from {raw_dir.resolve()}...")
    
    # 1. Load Calendar data
    calendar_df = pd.read_csv(calendar_path)
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
    # 2. Load Prices data
    prices_df = pd.read_csv(prices_path)
    
    # 3. Load Sales data
    sales_df = pd.read_csv(sales_path)
    
    # Filter target stores / items if defined in config for rapid iterative workflow
    sample_stores = config["data"].get("sample_stores", None)
    if sample_stores:
        sales_df = sales_df[sales_df['store_id'].isin(sample_stores)].reset_index(drop=True)
        logger.info(f"Filtered sales dataset to stores: {sample_stores}")
        
    max_items = config["data"].get("max_items_per_store", None)
    if max_items and max_items > 0:
        top_items = sales_df.groupby('item_id')['d_1'].count().head(max_items).index
        sales_df = sales_df[sales_df['item_id'].isin(top_items)].reset_index(drop=True)
        logger.info(f"Filtered sales dataset to top {max_items} items per store")
        
    # 4. Melt wide sales columns (d_1 to d_N) into long format
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = [c for c in sales_df.columns if c.startswith('d_')]
    
    logger.info(f"Reshaping sales table from wide to long format ({len(sales_df):,} series x {len(d_cols)} days)...")
    sales_long = sales_df.melt(id_vars=id_vars, value_vars=d_cols, var_name='d', value_name='sales')
    
    # Extract numeric day index for sorting
    sales_long['d_num'] = sales_long['d'].apply(lambda x: int(x.split('_')[1])).astype(np.int16)
    
    # 5. Join Calendar table
    logger.info("Joining calendar metadata...")
    sales_long = sales_long.merge(calendar_df, on='d', how='left')
    
    # 6. Join Sell Prices table
    logger.info("Joining price metadata...")
    sales_long = sales_long.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    # Impute missing sell prices using group mean / ffill per item-store
    if sales_long['sell_price'].isnull().sum() > 0:
        sales_long['sell_price'] = sales_long.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.ffill().bfill())
        # If any remain (item never priced), fill with overall item mean
        sales_long['sell_price'] = sales_long.groupby('item_id')['sell_price'].transform(lambda x: x.fillna(x.mean()))
        sales_long['sell_price'].fillna(sales_long['sell_price'].mean(), inplace=True)
    
    # 7. Execute Data Quality Checks
    quality_checks = perform_data_quality_checks(sales_long)
    
    # 8. Memory Optimization & Export
    sales_long = reduce_mem_usage(sales_long)
    
    output_parquet = processed_dir / "sales_long.parquet"
    output_csv = processed_dir / "sales_long_sample.csv"
    
    logger.info(f"Saving processed dataset to {output_parquet}...")
    sales_long.to_parquet(output_parquet, index=False)
    sales_long.head(1000).to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    logger.info(f"Ingestion completed successfully in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
    return sales_long, quality_checks


if __name__ == "__main__":
    run_ingestion()
