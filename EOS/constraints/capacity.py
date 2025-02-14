import numpy as np
import pandas as pd


def getFeasibilityMatrix(req_feasible: pd.DataFrame, cam_resolution: int = 1, compression_factor: int = 2, **kwargs) -> np.ndarray:
    K = (req_feasible["area"] * (1 / cam_resolution)) / compression_factor
    K = np.array(K).reshape((1, len(K)))

    return K
