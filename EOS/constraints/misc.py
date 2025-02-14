import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta


# Maximum one attempt per timestep per satellite
def getFeasibilityMatrixAcquisitions(req_feasible: pd.DataFrame, t_acq: list[timedelta,], **kwargs) -> np.ndarray:
    A = []
    for i, (sat_name, requests) in enumerate(tqdm(req_feasible.groupby(["sat_name"]), desc="Creating Single Attempt Constraint Matrix")):
        mat = np.zeros((len(t_acq), len(req_feasible)))
        for j, t in enumerate(requests["acq_time"]):
            t_idx = t_acq.index(t)  # Find the first index of the acquisition time
            mat[t_idx, requests.index[j]] = 1
        A.append(mat)
    constraint_matrix_A = np.concatenate(A, axis=0)
    constraint_matrix_A = constraint_matrix_A[np.sum(constraint_matrix_A, axis=1) != 0, :]  # time

    return constraint_matrix_A


# Maximum h attempt per request
def getFeasbilityMatrixCombined(req_feasible: pd.DataFrame, **kwargs) -> np.ndarray:
    req_unique_latlon = req_feasible["req_latlon"].unique()
    B = np.zeros(shape=(len(req_unique_latlon), len(req_feasible)))
    B_rhs = np.ones(len(req_unique_latlon))
    for i, pos in enumerate(tqdm(req_unique_latlon, desc="Creating h Attempt Constraint Matrix")):
        mask = (req_feasible["req_latlon"] == pos).to_numpy()
        B[i, :] = mask
        idx_first = np.where(mask)[0][0]
        B_rhs[i] = max(req_feasible["stereo"][idx_first] + 1, req_feasible["strips"][idx_first])
    return B, B_rhs


def getFeasibilityMatrixAngles(req_feasible: pd.DataFrame, angular_velocity: float, **kwargs) -> np.ndarray:
    FF = np.identity(len(req_feasible))
    for i, (sat_name, requests) in enumerate(tqdm(req_feasible.groupby(["sat_name"]), desc="Creating Rotational Contraint Matrix")):
        for idx_i, row_i in tqdm(list(requests.iterrows()), postfix=f"Satellite {sat_name[0]}"):
            vec_i = row_i["sat_xyz"] - row_i["req_xyz"]
            vec_i = vec_i / np.linalg.norm(vec_i, ord=2)

            req_finished = row_i["acq_time"] + timedelta(seconds=row_i["duration"])  # Timestamp for when the request has been processed
            feasible = requests[requests["acq_time"] > req_finished]  # Only consider acquisitions after the request has been processed
            feasible_r = list(feasible.sort_values(by="acq_time", ascending=False).iterrows())  # Reverse order to get the latest acquisitions first
            for j, (idx_j, row_j) in enumerate(feasible_r):
                vec_j = row_j["sat_xyz"] - row_j["req_xyz"]
                vec_j = vec_j / np.linalg.norm(vec_j, ord=2)

                angle = np.degrees(np.acos(vec_i @ vec_j))
                t_turn = timedelta(seconds=angle / angular_velocity)
                if t_turn + req_finished > row_j["acq_time"]:
                    idx_infeasible = [idx for idx, _ in feasible_r[j:]]
                    FF[idx_i, idx_infeasible] = 1
                    break
    return FF
