from models import ridge_model, lasso_model, random_forest_model, deep_model, gbm_model

MODEL_REGISTRY = {
    "ridge": ridge_model,
    "lasso": lasso_model,
    "random forest": random_forest_model,
    "gbm": gbm_model,
    "mlp": deep_model
}
