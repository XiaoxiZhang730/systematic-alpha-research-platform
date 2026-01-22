import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()
    
class MLPWrapper:
    def __init__(self, model, scaler):
        self.model = model.eval()
        self.scaler = scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()
        return preds

def fit_and_predict(X_train, y_train, X_test, cfg):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = y_train.values

    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MLP(X_train.shape[1], cfg.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    # wrap and predict
    wrapped_model = MLPWrapper(model, scaler)
    y_pred = wrapped_model.predict(X_test)

    return y_pred, wrapped_model


# def fit_and_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, cfg: Any) -> Tuple[np.ndarray, Any]:
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

#     model = MLP(input_dim=X_train.shape[1], hidden_dim=cfg.hidden_dim)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     X_train_tensor = X_train_tensor.to(device)
#     y_train_tensor = y_train_tensor.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
#     loss_fn = nn.MSELoss()

#     model.train()
#     for epoch in range(cfg.epochs):
#         optimizer.zero_grad()
#         preds = model(X_train_tensor)
#         loss = loss_fn(preds, y_train_tensor)
#         loss.backward()
#         optimizer.step()

#     # Predict
#     model.eval()
#     with torch.no_grad():
#         X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
#         preds = model(X_test_tensor).cpu().numpy()

#     # Return model object as a tuple with scaler + model (since not a pipeline)
#     return preds, model
