import time
from datetime import timedelta
import numpy as np
from cvxopt import glpk, matrix


def solve(score, bin_vars, LHS, RHS, eLHS, eRHS, **kwargs):

    t_start = time.perf_counter()

    # Objective coefficients (assuming score is a list or numpy array of coefficients)
    f = -np.array(score)

    # Convert numpy arrays to cvxopt matrices
    c = matrix(f)
    G = matrix(LHS)
    h = matrix(RHS)
    A = matrix(eLHS) if eLHS.size else None
    b = matrix(eRHS) if eRHS.size else None

    # Solve the problem using GLPK
    status, x = glpk.ilp(c=c, G=G, h=h, A=A, b=b, I=set(bin_vars), B=set(bin_vars))

    # Extract the solution
    x = np.array(x).flatten()

    print(f"Optimization time: {timedelta(time.perf_counter() - t_start)}")
    return np.squeeze(x)
