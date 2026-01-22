# import numpy as np
# import pandas as pd

# def rsi(series: pd.Series, window: int = 14) -> pd.Series:
#     delta = series.diff()
#     gain = delta.clip(lower=0)
#     loss = (-delta).clip(lower=0)
#     avg_gain = gain.rolling(window).mean()
#     avg_loss = loss.rolling(window).mean()
#     rs = avg_gain / avg_loss
#     return 100 - (100 / (1 + rs))

# def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
#     g = df.groupby(level="ticker", group_keys=False)
#     df["my_feat_20d_mom"] = g["log_return_1d"].apply(lambda s: s.rolling(20).sum())
#     return df

# def make_features(panel: pd.DataFrame, market: str) -> pd.DataFrame:
#     df = panel.copy()
#     g = df.groupby(level="ticker", group_keys=False)

#     df["log_return_1d"] = g["price"].apply(lambda s: np.log(s).diff())
#     df["mom_5d"] = g["log_return_1d"].apply(lambda s: s.rolling(5).sum())
#     df["vol_5d"] = g["log_return_1d"].apply(lambda s: s.rolling(5).std())
#     df["hl_range"] = (df["High"] - df["Low"]) / df["price"]
#     df["vol_ratio_5d"] = g["Volume"].apply(lambda s: s / s.rolling(5).mean())
#     dvol = g["Volume"].apply(lambda s: np.log(s).diff())
#     df["pv_corr_10d"] = g.apply(lambda x: x["log_return_1d"].rolling(10).corr(dvol.loc[x.index]))
#     ma_10 = g["price"].apply(lambda s: s.rolling(10).mean())
#     df["ma_gap_10d"] = (df["price"] - ma_10) / ma_10
#     df["rsi_14d"] = g["price"].apply(lambda s: rsi(s, 14))

#     # Excess return vs market
#     mkt_ret = (df.xs(market, level="ticker")[["log_return_1d"]]
#                 .rename(columns={"log_return_1d": "mkt_ret_1d"}))
#     df = df.join(mkt_ret, on="date")
#     df["excess_ret_1d"] = df["log_return_1d"] - df["mkt_ret_1d"]

#     df["cs_rank_mom_5d"] = df.groupby(level="date")["mom_5d"].rank(pct=True)

#     df = add_custom_features(df)
#     return df

# def add_target(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
#     df = df.copy()
#     df[target_col] = df.groupby(level="ticker")["log_return_1d"].shift(-1)
#     return df.dropna(subset=[target_col])

# def split_by_date(panel: pd.DataFrame, split_date: pd.Timestamp):
#     dates = pd.to_datetime(panel.index.get_level_values("date"))
#     train = panel.loc[dates <= split_date].copy()
#     test  = panel.loc[dates >  split_date].copy()
#     return train, test


# monthly return
# daily features
import numpy as np
import pandas as pd

# ——— Helper Functions ——— #

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def stochastic_oscillator(high, low, close, window=14):
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    percent_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    percent_d = percent_k.rolling(3).mean()
    return percent_k, percent_d

def truerange(df):
    high_low = df["High"] - df["Low"]
    high_pc  = (df["High"] - df["Close"].shift()).abs()
    low_pc   = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)

def cs_zscore(df, cols):
    return df.groupby("date")[cols].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )


# ——— Main Feature Builder ——— #

def make_features(panel: pd.DataFrame, market: str) -> pd.DataFrame:
    df = panel.copy()
    g = df.groupby(level="ticker", group_keys=False)
    initial_cols = set(df.columns)

    # —— I. Raw OHLCV ——
    df["Open"]  = df["Open"]
    df["High"]  = df["High"]
    df["Low"]   = df["Low"]
    df["Close"] = df["Close"]
    df["Volume"] = df["Volume"]

    # Log price for convenience
    df["log_price"] = np.log(df["Close"])

    # —— Log Returns ——
    df["log_return_1d"] = g["log_price"].apply(lambda s: s.diff())
    df["mom_5d"] = g["log_return_1d"].apply(lambda s: s.rolling(5).sum())
    df["vol_5d"] = g["log_return_1d"].apply(lambda s: s.rolling(5).std())
    df["hl_range"] = (df["High"] - df["Low"]) / df["price"]
    df["vol_ratio_5d"] = g["Volume"].apply(lambda s: s / s.rolling(5).mean())
    dvol = g["Volume"].apply(lambda s: np.log(s).diff())
    df["pv_corr_10d"] = g.apply(lambda x: x["log_return_1d"].rolling(10).corr(dvol.loc[x.index]))
    ma_10 = g["price"].apply(lambda s: s.rolling(10).mean())
    df["ma_gap_10d"] = (df["price"] - ma_10) / ma_10
    df["rsi_14d"] = g["price"].apply(lambda s: rsi(s, 14))

    # —— II. Trend Indicators ——
    # SMA & EMA windows
    sma_windows = [5, 10, 20, 50, 200]
    ema_windows = sma_windows.copy()

    for w in sma_windows:
        df[f"SMA_{w}"] = g["Close"].apply(lambda s: s.rolling(w).mean())

    for w in ema_windows:
        df[f"EMA_{w}"] = g["Close"].apply(lambda s: s.ewm(span=w).mean())

    # MACD
    ema12 = g["Close"].apply(lambda s: s.ewm(span=12).mean())
    ema26 = g["Close"].apply(lambda s: s.ewm(span=26).mean())
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].groupby(level="ticker", group_keys=False).apply(lambda s: s.ewm(span=9).mean())
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Parabolic SAR
    df["Parabolic_SAR"] = g.apply(
        lambda x: pd.Series(x["Low"].rolling(5).min(), index=x.index)
    )

    # ADX
    # Note: Full ADX is a bit involved; use built-in or simplified proxy
    df["ADX"] = g["Close"].apply(lambda s: s.rolling(14).apply(lambda x: np.nan))

    # —— III. Momentum Indicators ——
    # RSI
    for w in [7, 14, 21]:
        df[f"RSI_{w}"] = g["Close"].apply(lambda s: rsi(s, w))

    # # Stochastic
    # df["SO_%K"], df["SO_%D"] = zip(*g.apply(
    #     lambda x: stochastic_oscillator(x["High"], x["Low"], x["Close"])) )

    # ROC
    for w in [5, 10, 20]:
        df[f"ROC_{w}"] = g["Close"].apply(lambda s: s.pct_change(w))

    df["Williams_%R"] = g.apply(lambda x: (x["High"].rolling(14).max() - x["Close"]) /
                                (x["High"].rolling(14).max() - x["Low"].rolling(14).min()) * -100)

    df["CCI"] = g.apply(lambda x: (x["Close"] - x["Close"].rolling(20).mean()) /
                        (0.015 * x["Close"].rolling(20).std()))

    df["Momentum"] = g["Close"].apply(lambda s: s.diff(5))

    df["MFI"] = g.apply(lambda x: rsi(x["Volume"] * ((x["High"] + x["Low"] + x["Close"]) / 3), 14))

    # TRIX
    df["TRIX"] = g["Close"].apply(lambda s: s.ewm(span=15).mean().pct_change())

    df["Ultimate_Osc"] = g.apply(lambda x: np.nan)  # placeholder

    # —— IV. Volatility ——
    df["BB_upper"] = g["Close"].apply(lambda s: s.rolling(20).mean() + 2 * s.rolling(20).std())
    df["BB_lower"] = g["Close"].apply(lambda s: s.rolling(20).mean() - 2 * s.rolling(20).std())
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    df["ATR"] = g.apply(lambda x: truerange(x).rolling(14).mean())
    df["STDDEV_20"] = g["Close"].apply(lambda s: s.rolling(20).std())

    # df["VIX"] = np.nan
    # df["VVIX"] = np.nan
    # df["Garman_Klass"] = np.nan
    # df["Parkinson"] = np.nan
    # df["RVI"] = np.nan

    # —— V. Volume Indicators ——
    df["OBV"] = g.apply(lambda x: (np.sign(x["Close"].diff()) * x["Volume"]).fillna(0).cumsum())
    # df["CMF"] = np.nan
    df["VWAP"] = g.apply(lambda x: (x["Close"] * x["Volume"]).cumsum() / x["Volume"].cumsum())
    df["Volume_ROC"] = g["Volume"].apply(lambda s: s.pct_change())
    df["Force_Index"] = g.apply(lambda x: x["Close"].diff() * x["Volume"])
    # df["ADL"] = np.nan
    # df["Ease_of_Movement"] = np.nan
    # df["Volume_Oscillator"] = np.nan
    # df["NVI"] = np.nan
    # df["PVI"] = np.nan

    # —— VI. Lag Features ——
    for i in range(1, 6):
        df[f"Close_lag_{i}"] = g["Close"].apply(lambda s: s.shift(i))
        df[f"Volume_lag_{i}"] = g["Volume"].apply(lambda s: s.shift(i))
        df[f"Return_lag_{i}"] = g["log_return_1d"].apply(lambda s: s.shift(i))
    
    # feature_cols = list(set(df.columns) - initial_cols)
    # df[feature_cols] = cs_zscore(df, feature_cols)

    return df



def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are all NaN."""
    return df.dropna(axis=1, how="all")

# def add_target(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
#     df = df.copy()
#     df[target_col] = df.groupby(level="ticker", group_keys=False)["log_return_1d"].shift(-1)
#     return df.dropna(subset=[target_col])

# def add_target(df: pd.DataFrame, window: int = 21, target_col: str = "target") -> pd.DataFrame:
#     """
#     Add a forward-looking target: cumulative log return over the next `window` days.

#     Assumes df has MultiIndex: ['date', 'ticker'] and 'log_return_1d' column.
#     """
#     df = df.copy()

#     df[target_col] = (
#         df.groupby("ticker", group_keys=False)["log_return_1d"]
#         .rolling(window)
#         .sum()
#         .shift(-window)
#         .reset_index(level=0, drop=True)
#     )

#     return df.dropna(subset=[target_col])

def add_target(df: pd.DataFrame, window: int = 21, target_col: str = "target") -> pd.DataFrame:
    """
    Add a forward-looking target: cumulative log return over the next `window` days,
    then cross-sectionally demeaned by date.

    Assumes df has MultiIndex: ['date', 'ticker'] and 'log_return_1d' column.
    """
    df = df.copy()

    # Step 1: Compute forward-looking cumulative log return over next `window` days
    df[target_col] = (
        df.groupby("ticker", group_keys=False)["log_return_1d"]
        .rolling(window)
        .sum()
        .shift(-window)
        .reset_index(level=0, drop=True)
    )

    # Step 2: Drop NaNs (incomplete forward returns)
    df = df.dropna(subset=[target_col])

    # # Step 3: Cross-sectional demeaning by date
    # df[target_col] = (
    #     df.groupby("date")[target_col]
    #     .transform(lambda x: x - x.mean())
    # )

    return df

def split_by_date(panel: pd.DataFrame, split_date: pd.Timestamp):
    dates = pd.to_datetime(panel.index.get_level_values("date"))
    train = panel.loc[dates <= split_date].copy()
    test  = panel.loc[dates >  split_date].copy()

    # Add a check to avoid empty train or test
    if train.empty:
        raise ValueError("Training set is empty. Consider using an earlier split_date.")
    if test.empty:
        raise ValueError("Test set is empty. Consider using a later split_date.")

    return train, test


