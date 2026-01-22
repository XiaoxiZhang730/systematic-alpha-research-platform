import json
import hashlib
from datetime import datetime
from typing import List, Any

def hash_config(cfg: Any) -> str:
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__annotations__}
    cfg_json = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.md5(cfg_json.encode()).hexdigest()

def hash_features(features: List[str]) -> str:
    feat_str = ",".join(sorted(features))
    return hashlib.sha256(feat_str.encode()).hexdigest()

def generate_run_id(cfg: Any, features: List[str], model_name: str = "ridge") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = hash_config(cfg)
    feat_hash = hash_features(features)
    return f"{timestamp}_{model_name}_{cfg_hash[:6]}_{feat_hash[:6]}"

