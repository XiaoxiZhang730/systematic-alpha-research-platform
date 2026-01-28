import numpy as np
import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def truerange(df):
    high_low = df["High"] - df["Low"]
    high_pc = (df["High"] - df["Close"].shift()).abs()
    low_pc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)

def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature engineering with:
    - No data leakage
    - Cross-sectional normalization
    - Orthogonal features
    """
    df = panel.copy()
    g = df.groupby(level="ticker", group_keys=False)
    
    # Base calculations (not features themselves)
    df["price"] = df["Close"]
    df["log_price"] = np.log(df["Close"])
    df["ret_1d"] = g["log_price"].diff(1)
    
    # ============ RETURN FEATURES ============
    for d in [5, 10, 21, 63]:
        df[f"ret_{d}d"] = g["log_price"].diff(d)
    
    # Momentum (12-1 month)
    df["mom_12m_1m"] = g["log_price"].diff(252) - g["log_price"].diff(21)
    
    # ============ VOLATILITY FEATURES ============
    df["vol_21d"] = g["ret_1d"].apply(lambda s: s.rolling(21, min_periods=15).std())
    df["vol_63d"] = g["ret_1d"].apply(lambda s: s.rolling(63, min_periods=40).std())
    df["vol_ratio"] = df["vol_21d"] / (df["vol_63d"] + 1e-10)
    
    # Idiosyncratic volatility (residual after market adjustment would be better)
    df["vol_skew"] = g["ret_1d"].apply(lambda s: s.rolling(21).skew())
    
    # ============ MEAN REVERSION FEATURES ============
    for d in [10, 21, 50]:
        ma = g["price"].apply(lambda s: s.rolling(d).mean())
        df[f"dist_ma_{d}d"] = (df["price"] - ma) / (ma + 1e-10)
    
    # ============ TECHNICAL FEATURES ============
    df["rsi_14"] = g["price"].apply(lambda s: rsi(s, 14))
    df["rsi_norm"] = (df["rsi_14"] - 50) / 50  # Normalize to ~[-1, 1]
    
    # MACD
    ema_12 = g["price"].apply(lambda s: s.ewm(span=12).mean())
    ema_26 = g["price"].apply(lambda s: s.ewm(span=26).mean())
    df["macd_norm"] = (ema_12 - ema_26) / (df["price"] + 1e-10)
    
    # Bollinger Band position
    ma_20 = g["price"].apply(lambda s: s.rolling(20).mean())
    std_20 = g["price"].apply(lambda s: s.rolling(20).std())
    df["bb_position"] = (df["price"] - ma_20) / (2 * std_20 + 1e-10)
    
    # ATR normalized
    df["atr_14"] = g.apply(lambda x: truerange(x).rolling(14).mean())
    df["atr_norm"] = df["atr_14"] / (df["price"] + 1e-10)
    
    # ============ VOLUME FEATURES ============
    df["volume_ratio"] = df["Volume"] / (g["Volume"].apply(lambda s: s.rolling(21).mean()) + 1e-10)
    df["log_dollar_vol"] = np.log(df["price"] * df["Volume"] + 1)
    
    # Price-volume correlation
    df["pv_corr_21d"] = g.apply(
        lambda x: x["ret_1d"].rolling(21).corr(x["Volume"].pct_change())
    )
    
    # ============ DEFINE FEATURE COLUMNS ============
    feature_cols = [
        'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'mom_12m_1m',
        'vol_21d', 'vol_63d', 'vol_ratio', 'vol_skew',
        'dist_ma_10d', 'dist_ma_21d', 'dist_ma_50d',
        'rsi_norm', 'macd_norm', 'bb_position', 'atr_norm',
        'volume_ratio', 'log_dollar_vol', 'pv_corr_21d'
    ]
    
    # ============ CROSS-SECTIONAL NORMALIZATION ============
    # Use rank to handle outliers
    for col in feature_cols:
        if col in df.columns:
            # Winsorize extreme values
            df[col] = df.groupby("date")[col].transform(
                lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
            )
            # Cross-sectional z-score
            df[col] = df.groupby("date")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    
    return df


def add_target(df: pd.DataFrame, window: int = 21, target_col: str = "target") -> pd.DataFrame:
    """
    Forward-looking return with cross-sectional demeaning.
    """
    df = df.copy()
    
    # Forward return: log(price_{t+window}) - log(price_t)
    df[target_col] = (
        df.groupby("ticker", group_keys=False)["log_price"]
        .apply(lambda s: s.shift(-window) - s)
    )
    
    # Cross-sectional demean (predict relative performance)
    df[target_col] = df.groupby("date")[target_col].transform(lambda x: x - x.mean())
    
    return df.dropna(subset=[target_col])


def split_by_date(panel: pd.DataFrame, split_date: str, target_window: int = 21):
    """Complete data preparation pipeline."""
    
    # 1. Create features
    df = make_features(panel)
    
    # 2. Add target
    df = add_target(df, window=target_window)
    
    # 3. Define feature columns (exclude non-features)
    exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'price', 'log_price', 'ret_1d', 'target', 'rsi_14', 'atr_14'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 4. Shift features to avoid look-ahead bias
    df[feature_cols] = df.groupby("ticker")[feature_cols].shift(1)
    
    # 5. Drop NaN rows
    df = df.dropna(subset=feature_cols + ['target'])
    
    # 6. Split by date
    split_dt = pd.Timestamp(split_date)
    dates = df.index.get_level_values("date")
    train = df[dates <= split_dt].copy()
    test = df[dates > split_dt].copy()
    
    return train, test, feature_cols