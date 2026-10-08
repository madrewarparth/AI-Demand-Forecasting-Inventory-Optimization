"""
Module: Model Training & Experimentation Pipeline
Trains baseline models (Naive / Moving Average) and LightGBM Regressor for 28-day demand forecasting.
Tracks experiments, metrics (RMSE, MAE, WRMSSE), parameters, and artifacts via MLflow.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_wrmsse(y_true, y_pred, y_hist, weights=None):
    """
    Weighted Root Mean Squared Scaled Error (WRMSSE) metric for M5 time series.
    """
    # Scale denominator: mean squared differences of historical consecutive days
    scale = np.mean(np.diff(y_hist) ** 2)
    if scale == 0 or np.isnan(scale):
        scale = 1.0
        
    rmsse = np.sqrt(mean_squared_error(y_true, y_pred) / scale)
    return float(rmsse)


def evaluate_forecasts(test_df, pred_col, label_col='sales', hist_df=None):
    """Computes RMSE, MAE, and estimated WRMSSE for a forecast column."""
    y_true = test_df[label_col].values
    y_pred = test_df[pred_col].values
    
    # Clip negative predictions to 0
    y_pred = np.clip(y_pred, 0, None)
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # Calculate average series-level RMSSE / WRMSSE
    series_wrmsses = []
    for series_id, group in test_df.groupby('id'):
        if len(group) > 0:
            hist_sales = hist_df[hist_df['id'] == series_id]['sales'].values if hist_df is not None else group['sales'].values
            s_wrmsse = calculate_wrmsse(group[label_col].values, group[pred_col].values, hist_sales)
            series_wrmsses.append(s_wrmsse)
            
    wrmsse = float(np.mean(series_wrmsses)) if series_wrmsses else rmse
    
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "wrmsse": round(wrmsse, 4)}


def run_training(config_path="configs/config.yaml"):
    start_time = time.time()
    config = load_config(config_path)
    
    processed_dir = Path(config["paths"]["processed_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    input_parquet = processed_dir / "train_features.parquet"
    logger.info(f"Loading feature-engineered dataset from {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    
    # Filter out initial lag warmup period (first 42 days)
    df = df[df['d_num'] > 42].reset_index(drop=True)
    
    # Determine time-based holdout split
    max_d = df['d_num'].max()
    horizon = config["data"].get("forecast_horizon", 28)
    split_d = max_d - horizon
    
    logger.info(f"Holdout Split: Train (d_43 to d_{split_d}), Test/Holdout (d_{split_d+1} to d_{max_d}) [{horizon} days]")
    
    train_df = df[df['d_num'] <= split_d].reset_index(drop=True)
    test_df = df[df['d_num'] > split_d].copy().reset_index(drop=True)
    
    # 1. Baseline Model (Naive Moving Average: 28-day rolling mean & lag_28)
    logger.info("Computing Baseline predictions (Naive / Moving Average)...")
    test_df['pred_baseline'] = test_df['rolling_mean_28'].fillna(test_df['lag_28']).fillna(0)
    baseline_metrics = evaluate_forecasts(test_df, pred_col='pred_baseline', label_col='sales', hist_df=train_df)
    logger.info(f"Baseline Metrics -> RMSE: {baseline_metrics['rmse']}, MAE: {baseline_metrics['mae']}, WRMSSE: {baseline_metrics['wrmsse']}")
    
    # 2. LightGBM Forecasting Model
    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'wday', 'month', 'year', 'day_of_month', 'week_of_year', 'is_weekend', 'snap_active',
        'event_name_1', 'event_type_1', 'has_event',
        'sell_price', 'price_rel_mean', 'price_discount_ratio',
        'lag_7', 'lag_14', 'lag_21', 'lag_28', 'lag_35', 'lag_42',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
        'rolling_mean_28', 'rolling_std_28', 'rolling_max_28', 'rolling_min_28'
    ]
    
    # Convert categorical feature columns
    cat_cols = [c for c in features if df[c].dtype.name == 'category' or df[c].dtype == object]
    for c in cat_cols:
        train_df[c] = train_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')
        
    X_train, y_train = train_df[features], train_df['sales']
    X_test, y_test = test_df[features], test_df['sales']
    
    logger.info(f"Training LightGBM model on {len(X_train):,} training samples with {len(features)} features...")
    lgb_params = config["model"]["lgb_params"]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # Predict LightGBM
    test_df['pred_lightgbm'] = np.clip(model.predict(X_test), 0, None)
    lgb_metrics = evaluate_forecasts(test_df, pred_col='pred_lightgbm', label_col='sales', hist_df=train_df)
    logger.info(f"LightGBM Metrics -> RMSE: {lgb_metrics['rmse']}, MAE: {lgb_metrics['mae']}, WRMSSE: {lgb_metrics['wrmsse']}")
    
    # Calculate percentage improvements
    rmse_imp = round(((baseline_metrics['rmse'] - lgb_metrics['rmse']) / baseline_metrics['rmse']) * 100, 2)
    mae_imp = round(((baseline_metrics['mae'] - lgb_metrics['mae']) / baseline_metrics['mae']) * 100, 2)
    wrmsse_imp = round(((baseline_metrics['wrmsse'] - lgb_metrics['wrmsse']) / baseline_metrics['wrmsse']) * 100, 2)
    
    logger.info(f"LightGBM Improvement over Baseline -> RMSE: +{rmse_imp}%, MAE: +{mae_imp}%, WRMSSE: +{wrmsse_imp}%")
    
    # Extract Feature Importances
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    top_features = importances.head(10).to_dict(orient='records')
    
    # 3. Log Experiment to MLflow
    mlflow.set_experiment("M5_Demand_Forecasting")
    with mlflow.start_run(run_name="LightGBM_Demand_Model"):
        mlflow.log_params(lgb_params)
        mlflow.log_metrics({
            "baseline_rmse": baseline_metrics['rmse'],
            "baseline_mae": baseline_metrics['mae'],
            "baseline_wrmsse": baseline_metrics['wrmsse'],
            "lgb_rmse": lgb_metrics['rmse'],
            "lgb_mae": lgb_metrics['mae'],
            "lgb_wrmsse": lgb_metrics['wrmsse'],
            "rmse_improvement_pct": rmse_imp,
            "mae_improvement_pct": mae_imp,
            "wrmsse_improvement_pct": wrmsse_imp
        })
        
    # 4. Save Model & Forecast Outputs
    import joblib
    model_path = models_dir / "lightgbm_demand_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved trained LightGBM model artifact to {model_path}")
    
    forecasts_path = processed_dir / "forecasts.csv"
    forecast_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'date', 'd', 'd_num', 'sales', 'pred_baseline', 'pred_lightgbm', 'sell_price']
    test_df[forecast_cols].to_csv(forecasts_path, index=False)
    logger.info(f"Exported holdout 28-day forecasts to {forecasts_path}")
    
    elapsed_minutes = round((time.time() - start_time) / 60, 2)
    
    summary = {
        "baseline_metrics": baseline_metrics,
        "lightgbm_metrics": lgb_metrics,
        "improvements": {
            "rmse_improvement_pct": rmse_imp,
            "mae_improvement_pct": mae_imp,
            "wrmsse_improvement_pct": wrmsse_imp
        },
        "top_features": top_features,
        "pipeline_execution_time_minutes": elapsed_minutes,
        "total_series_processed": int(test_df['id'].nunique())
    }
    
    summary_path = models_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"Training pipeline completed successfully in {elapsed_minutes} minutes.")
    return summary


if __name__ == "__main__":
    run_training()
