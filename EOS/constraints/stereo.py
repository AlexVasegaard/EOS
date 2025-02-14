import numpy as np
from tqdm import tqdm
import pandas as pd


def getFeasiblePairs(req_feasible: pd.DataFrame, stereo_angle: float = 15, stereo_error: float = 5, **kwargs) -> np.ndarray:
    # First we check the angle requirements for the stereo images
    stereo_proj_min = np.cos(np.radians(stereo_angle - stereo_error))  # Ensures we don't have to calculate acos for every pair
    stereo_proj_max = np.cos(np.radians(stereo_angle + stereo_error))  # because we can't use build in angle in req_feasible
    # stereo_angle = 15, stereo_error=2.5 => 12.5-17.5 degrees

    feasible_pais = []
    req_stereo = req_feasible.loc[req_feasible["stereo"] == True].reset_index(drop=True)
    for req_id, group in tqdm(req_stereo.groupby("ID"), desc="Computing Feasible Stereo Pairs"):
        if len(group) == 1:
            continue

        group = group.sort_values(by="acq_time")
        for i, (idx_i, row_i) in enumerate(group.iterrows()):
            vec_i = row_i["sat_xyz"] - row_i["req_xyz"]
            vec_i = vec_i / np.linalg.norm(vec_i, ord=2)
            for j, (idx_j, row_j) in enumerate(group.iloc[i + 1 :].iterrows()):
                vec_j = row_j["sat_xyz"] - row_j["req_xyz"]  # row_j["req_xyz"] is the same as row_i["req_xyz"]
                vec_j = vec_j / np.linalg.norm(vec_j, ord=2)

                proj = vec_i @ vec_j
                if stereo_proj_max < proj < stereo_proj_min:  # Large angle yields small dot product, and vice versa
                    feasible_pais.append((idx_i, idx_j))

    return feasible_pais


def getFeasibilityMatrix(req_feasible: pd.DataFrame, feasible_pais: list[tuple[int, int]], **kwargs) -> pd.DataFrame:

    # We need to ensure that every stereo request is acquired by at least two satellites, however, these pairs essentially correspond to a single decision. That means that if the feasible pair [1, 7] is taken, then we need to move the pair [1,13] in the matrix. If we don't do this, we force the solver to choose both [1, 7] and [1, 13], as the matrix-vector product need to sum to 0 everywhere for the Ax = b formulation. Consequently, we need to ensure that no two stereo pairs share an index in the stereo constraint matrix
    idx_used = set()  # Track repeated indices
    rows_new = []  # Store df rows that should be repeated => Allows single call to pd.concat

    constraint_matrix_stereo = np.zeros(shape=(len(feasible_pais), len(req_feasible)))
    for i, pair in enumerate(tqdm(feasible_pais, desc="Creating Stereo Constraint Matrix")):
        for idx in pair:
            if idx in idx_used:
                constraint_matrix_stereo = np.insert(constraint_matrix_stereo, -1, 0, axis=1)  # Append column
                constraint_matrix_stereo[i, -1] = 1
                rows_new.append(req_feasible.iloc[idx])
            else:
                constraint_matrix_stereo[i, idx] = 1
                idx_used.add(idx)

    req_feasible = pd.concat([req_feasible, pd.DataFrame(rows_new)], ignore_index=True).reset_index(drop=True)
    req_feasible = req_feasible.sort_values(by=["sat_name", "acq_time"]).reset_index(drop=True)

    return constraint_matrix_stereo, req_feasible
