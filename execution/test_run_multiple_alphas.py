import os
import sys
import pandas as pd
from dataclasses import replace
import copy

# Force working dir to script's grandparent folder (your project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
os.chdir(project_root)
sys.path.insert(0, project_root)

print("Forced Working Directory to:", os.getcwd())

from configs.config import Config
from run_experiment import run_experiment

# Define list of alphas to test

# Load base config
# base_cfg = Config(
#         tickers=(
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
#     "JPM", "BAC", "GS",
#     "XOM",
#     "JNJ", "UNH",
#     "PG", "KO", "MCD",
#     "CAT", "BA",
#     "META",
#     "SPY", "QQQ", "IWM"),
#         market="SPY",
#         start="2015-01-01",
#         end="2024-12-31",
#         split_date=pd.Timestamp("2021-12-31")
#     )

# base_cfg = Config(
#         tickers=(
#         # ===== Tech =====
#         "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
#         "META", "ORCL", "ADBE", "CRM", "INTC",
#         "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP"
#         ),
#         market="SPY",
#         start="2015-01-01",
#         end="2024-12-31",
#         split_date=pd.Timestamp("2021-12-31")
#     )

# results = []

# alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]

# for alpha in alpha_values:
#     cfg = replace(base_cfg, ridge_alpha=alpha)
#     print(f"\n>>> Running experiment for alpha = {alpha}")
#     result = run_experiment(cfg, model_name="ridge")
#     results.append({
#         "alpha": alpha,
#         "r2": result["metrics"]["r2"],
#         "mae": result["metrics"]["mae"],
#         "rmse": result["metrics"]["rmse"],
#         "rank_ic": result["rank_ic"]["mean_rank_ic"],
#         "run_id": result["run_id"]
#     })

# df = pd.DataFrame(results)
# print("\n=== Summary ===")
# print(df.sort_values(by="r2", ascending=False).to_string(index=False))

# 旧配置
old_cfg = Config(
    tickers=(
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "JPM", "BAC", "GS",
        "XOM",
        "JNJ", "UNH",
        "PG", "KO", "MCD",
        "CAT", "BA",
        "META",
        # "SPY",
          "QQQ", "IWM"
    ),
    market="SPY",
    start="2015-01-01",
    end="2024-12-31",
    split_date=pd.Timestamp("2021-12-31")
)

# 新配置
new_cfg = Config(
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

configs = [("old", old_cfg), ("new", new_cfg)]

results = []
alpha_values = [0.01]

for alpha in alpha_values:
    for config_name, cfg in configs:
        cfg_with_alpha = replace(cfg, ridge_alpha=alpha)
        print(f"\n>>> Running experiment for alpha = {alpha} | config = {config_name}")
        result = run_experiment(cfg_with_alpha, model_name="ridge")
        results.append({
            "config": config_name,
            "alpha": alpha,
            "r2": result["metrics"]["r2"],
            "mae": result["metrics"]["mae"],
            "rmse": result["metrics"]["rmse"],
            "rank_ic": result["rank_ic"]["mean_rank_ic"],
            "run_id": result["run_id"]
        })

# 整合并输出
df = pd.DataFrame(results)
print("\n=== Summary ===")
print(df.sort_values(by=["alpha", "config"]).to_string(index=False))