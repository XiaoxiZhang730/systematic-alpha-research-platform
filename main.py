import pandas as pd
from configs.config import Config
from execution.multi_model_runner import run_all_models 

if __name__ == "__main__":
    cfg = Config(
        tickers=(
        # ===== Tech =====
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "ORCL", "ADBE", "CRM", "INTC",
        "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP"
        ),
        market="SPY",
        start="2015-01-01",
        end="2024-12-31",
        split_date=pd.Timestamp("2021-12-31")
    )
    run_all_models(cfg)

# if __name__ == "__main__":
#     cfg = Config(
#         tickers=(
#         # ===== Tech =====
#         "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
#         "META", "ORCL", "ADBE", "CRM", "INTC",

#         # ===== Financials =====
#         "JPM", "BAC", "GS", "MS", "WFC",
#         "C", "BLK", "AXP",

#         # ===== Healthcare =====
#         "JNJ", "UNH", "PFE", "MRK", "ABBV",
#         "LLY", "TMO",

#         # ===== Consumer Staples =====
#         "PG", "KO", "PEP", "WMT", "COST",

#         # ===== Consumer Discretionary =====
#         "MCD", "HD", "NKE", "LOW", "SBUX",

#         # ===== Industrials =====
#         "CAT", "BA", "GE", "HON", "UPS",

#         # ===== Energy =====
#         "XOM", "CVX",

#         # ===== Communication =====
#         "DIS", "NFLX",

#         # ===== Market / Style ETFs =====
#         "SPY", "QQQ", "IWM"
#         ),
#         market="SPY",
#         start="2015-01-01",
#         end="2024-12-31",
#         split_date=pd.Timestamp("2021-12-31")
#     )
#     run_all_models(cfg)
