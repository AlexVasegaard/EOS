from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from cvxopt import matrix, spmatrix
from scipy import sparse
from skyfield.sgp4lib import EarthSatellite
from tqdm import tqdm

from constraints import capacity as ConCapacity
from constraints import stereo as ConStereo
from constraints import strips as ConStrips
from constraints import misc as ConMisc


def scipy_sparse_to_spmatrix(A: np.ndarray) -> spmatrix:
    # Makes one of the solvers faster (GLPK?) if we convert the matrix to a spmatrix
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP


# LPP data computation - multi satellite (no battery, higher dim location_slots, etc)
# TODO: Make this a class
def createLPPFormulation(
    req_feasible: pd.DataFrame,  # Dataframe containing all feasible request-satellite-time combinations
    t_acq: list[datetime,],
    cam_resolution: float,
    angular_velocity=2.5,  # degrees per second
    stereo_angle: float = 15,
    stereo_error: float = 5,
    **kwargs,
):
    pbar = tqdm(total=9, desc="Formulating the LPP Problem")

    #### Stereo Requests ####
    feasible_pais = ConStereo.getFeasiblePairs(req_feasible, stereo_angle, stereo_error, **kwargs)
    pbar.update()

    constraint_matrix_stereo, req_feasible = ConStereo.getFeasibilityMatrix(req_feasible, feasible_pais, **kwargs)
    pbar.update()

    #### Maximum one attempt per timestep per satellite ####
    constraint_matrix_A = ConMisc.getFeasibilityMatrixAcquisitions(req_feasible, t_acq, **kwargs)
    pbar.update()

    #### Maximum h attempt per request ####
    B, B_rhs = ConMisc.getFeasbilityMatrixCombined(req_feasible, **kwargs)
    pbar.update()

    #### Rotational Constraints ####
    F = ConMisc.getFeasibilityMatrixAngles(req_feasible, angular_velocity, **kwargs)
    pbar.update()

    # TODO: COMMENT AND REWRITE THIS
    constraint_matrix_A = constraint_matrix_A[np.sum(constraint_matrix_A, axis=1) != 0, :]  # Time

    # Possibility of simplifying FF here!

    pbar.update()

    ## CAPACITY constraint
    # K = ConCapacity.getFeasibilityMatrix(req_feasible, cam_resolution, **kwargs)
    # pbar.update()

    # STRIPS constraint
    strips_constraint = ConStrips.getFeasbilityMatrix(req_feasible)
    pbar.update()

    #### OPTIMIZATION
    # less than constraint
    # LHS
    LESS_THAN_Matrix = np.concatenate((B, constraint_matrix_A, F), axis=0)  # (A_constraint)

    # RHS
    # one acq per request except for stereo -> it is there
    F_rhs = np.ones((F.shape[0] + constraint_matrix_A.shape[0], 1))  # added A
    rhs_ABF = np.concatenate((B_rhs, np.squeeze(F_rhs)), axis=0)

    # equal to constraint
    eLHS = constraint_matrix_stereo  # np.concatenate((S_constraint, Strips_constraint), axis = 0)
    eRHS = np.zeros(eLHS.shape[0])

    # drop empty rows
    non_empty_rows = ~(np.sum(LESS_THAN_Matrix, axis=1) == 0)
    LHS_leq = LESS_THAN_Matrix[non_empty_rows, :]
    RHS_leq = rhs_ABF[non_empty_rows]

    # convert LHS matrix to sparce matrix
    sparcity = np.sum(LHS_leq == 0) / (LHS_leq.shape[0] * LHS_leq.shape[1])  # level of sparsity
    if sparcity >= 0.75:
        b = sparse.csr_matrix(LHS_leq)  # Compressed sparse row matrix
        LHS_leq = scipy_sparse_to_spmatrix(b)
    if sparcity < 0.75:
        LHS_leq = matrix(LHS_leq)
    #################### LPP DONE ################################

    pbar.update()
    pbar.close()

    return {
        "B": B,
        "stereo": constraint_matrix_stereo,
        "strips": strips_constraint,
        "req_feasible": req_feasible,
        "F": F,
        "LHS": LHS_leq,
        "RHS": RHS_leq,
        "eLHS": eLHS,
        "eRHS": eRHS,
    }
