import time
import numpy as np
from cvxopt import matrix


def solve(score, LPP, len_p, N=1000):
    t_start = time.perf_counter()
    # generate 20000 random solutions and test objective function for each
    # choose interval that illustrates distribution of ones and zeros
    binary_int = [0.03, 0.18]
    binary_int_dist = binary_int[1] - binary_int[0]
    X_test = np.zeros((len_p, N))
    for i in range(0, N):
        dist_1 = binary_int[0] + i / N * (binary_int_dist)
        X_test[:, i] = np.random.choice([0, 1], size=(len_p,), p=[1 - dist_1, dist_1])

    # test if the solutions are valid
    valid_x = np.zeros((N))
    for i in range(0, N):
        if all(np.array(matrix(LPP.LHS)) @ X_test[:, i] <= LPP.RHS) and all(np.array(matrix(LPP.eLHS)) @ X_test[:, i] == LPP.eRHS):
            valid_x[i] = 1

    # find maximal score
    sum_valid = int(np.sum(valid_x))
    if sum_valid > 0:
        score_test = np.transpose(np.ones((sum_valid, len(score))) * score)
        score_sum = np.sum(X_test[:, np.where(valid_x)[0]] * score_test, axis=0)
        max_i = np.argmax(score_sum)
        x = X_test[:, np.where(valid_x)[0]][:, max_i]
        print(score_sum[max_i])
    else:
        x = np.zeros(len_p)
        print("Problem too complex!")

    t_end = time.perf_counter()
    print(f"Optimization time: {t_end - t_start}")
    return x
