from run_experiment import run_experiment
from typing import Any

def run_all_models(cfg: Any):
    model_names = ["ridge", "lasso", "random forest", "gbm"]
    #model_names = ["mlp"]
    results = []

    for model_name in model_names:
        print(f"\n🧪 Running model: {model_name}")
        try:
            result = run_experiment(cfg, model_name=model_name)
            results.append((model_name, result))
        except Exception as e:
            print(f"❌ Failed on model {model_name}: {e}")
            results.append((model_name, None))

    print("\n=== ✅ All Models Complete ===")
    print(f"{'Model':<10} {'R2':>6} {'MAE':>8} {'RMSE':>8} {'RankIC':>8} {'#Features':>10}")
    print("-" * 55)
    for model_name, output in results:
        if output is None:
            print(f"{model_name:<10} FAILED")
            continue
        r2 = output['metrics']['r2']
        mae = output['metrics']['mae']
        rmse = output['metrics']['rmse']
        ric = output['rank_ic']['mean_rank_ic']
        n_feat = len(output['used_features'])
        print(f"{model_name:<10} {r2:6.4f} {mae:8.4f} {rmse:8.4f} {ric:8.4f} {n_feat:10}")

# from run_experiment import run_experiment
# from typing import Any
# from dataclasses import replace

# def run_all_models(cfg: Any):
#     # model_names = ["ridge", "lasso", "random forest", "gbm"]
#     model_names = ["ridge"]
#     results = []

#     for model_name in model_names:
#         print(f"\n🧪 Running model: {model_name}")
#         try:
#             if model_name == "ridge":
#                 for alpha in cfg.alpha_grid:
#                     print(f"🔍 Trying alpha={alpha:.2e} for {model_name}")
#                     cfg_i =  replace(cfg, ridge_alpha=alpha)
#                     result = run_experiment(cfg_i, model_name=model_name)
#                     results.append((model_name, alpha, result))
#             elif model_name == "lasso":
#                 for alpha in cfg.alpha_grid:
#                     print(f"🔍 Trying alpha={alpha:.2e} for {model_name}")
#                     cfg_i =  replace(cfg, lasso_alpha=alpha)
#                     result = run_experiment(cfg_i, model_name=model_name)
#                     results.append((model_name, alpha, result))
#             else:
#                 result = run_experiment(cfg, model_name=model_name)
#                 results.append((model_name, None, result))
#         except Exception as e:
#             print(f"❌ Failed on model {model_name}: {e}")
#             results.append((model_name, None, None))

#     print("\n=== ✅ All Models Complete ===")
#     print(f"{'Model':<12} {'Alpha':>8} {'R2':>6} {'MAE':>8} {'RMSE':>8} {'RankIC':>8} {'#Features':>10}")
#     print("-" * 70)
#     for model_name, alpha, output in results:
#         if output is None:
#             print(f"{model_name:<12} {str(alpha):>8} FAILED")
#             continue
#         r2 = output['metrics']['r2']
#         mae = output['metrics']['mae']
#         rmse = output['metrics']['rmse']
#         ric = output['rank_ic']['mean_rank_ic']
#         n_feat = len(output['used_features'])
#         alpha_str = f"{alpha:.1e}" if alpha is not None else "N/A"
#         print(f"{model_name:<12} {alpha_str:>8} {r2:6.4f} {mae:8.4f} {rmse:8.4f} {ric:8.4f} {n_feat:10}")
