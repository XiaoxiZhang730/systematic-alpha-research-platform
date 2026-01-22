def json_safe(v):
    import numpy as np
    import pandas as pd

    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    elif isinstance(v, (frozenset, set)):
        return list(v)
    elif isinstance(v, (np.integer, int)):
        return int(v)
    elif isinstance(v, (np.floating, float)):
        return float(v)
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, pd.Series):
        return v.to_dict()
    elif isinstance(v, pd.DataFrame):
        return v.to_dict(orient="records")
    elif hasattr(v, '__dict__'):
        return v.__dict__
    return str(v)
