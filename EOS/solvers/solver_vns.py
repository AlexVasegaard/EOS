import numpy as np
from cvxopt import matrix
from datetime import datetime, timedelta


def solve(
    score,
    LPP,
    performance_df,
    max_time=120,
    max_iter_N=1000,
):
    # object fct
    c = score
    # <=
    G = np.array(matrix(LPP.LHS))
    h = LPP.RHS
    # =
    A = LPP.eLHS
    b = LPP.eRHS
    # number of runs

    # define neighbourhoods
    string_list = list()
    BBB = list(performance_df["request location"])
    string_list = list(set(list(map(str, BBB))))
    number_of_reach_areas = len(string_list)
    string_list_np = np.array([eval(n) for n in string_list])
    pfloc_np = np.array([xi for xi in performance_df["request location"]])
    N_i = np.zeros((number_of_reach_areas, len(pfloc_np)))
    N_i_rhs = np.zeros((number_of_reach_areas))
    for i in range(0, number_of_reach_areas):
        N_i[i, :] = np.sum((pfloc_np == string_list_np[i, :]), axis=1) == 2
        one_index_reach = np.where(np.sum(pfloc_np == string_list_np[i], axis=1) == 2)[0][0]
        N_i_rhs[i] = max(performance_df["stereo"][one_index_reach], performance_df["strips"][one_index_reach])

    # begin requestbased neighbourhood search

    VNS_time_max = datetime.now() + timedelta(seconds=max_time)
    fx_min = 0
    x_i = np.zeros((c.shape[0]))
    x_opt = np.zeros((c.shape[0]))
    N = 0
    while N < N_i.shape[0] and VNS_time_max > datetime.now():
        N_loc = np.where(N_i[N, :])[0]
        N_size = len(N_loc)
        x_i = list(x_opt)
        for i in range(0, max_iter_N):
            no_ones = np.random.randint(1, min(N_i_rhs[N], N_size))
            random_x = np.array([1] * no_ones + [0] * int(N_size - no_ones))
            np.random.shuffle(random_x)
            x_i = np.array(x_i)
            x_i[N_loc] = random_x
            fx_i = c @ x_i
            if (G @ x_i <= h).all() and (A @ x_i == b).all() and fx_i < fx_min:
                fx_min = fx_i
                x_opt = list(x_i)
                N = -1
                print(fx_min)
                break

    x = np.array(x_opt)
    return x
