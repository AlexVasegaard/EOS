import numpy as np
import pandas as pd
import pulp as plp
import time


def solve(score, RHS, LHS, eRHS, eLHS, **kwargs):

    # Initialize the model
    opt_model = plp.LpProblem(name="BLP_Model", sense=plp.LpMinimize)

    # Objective coefficients (assuming score is a list or numpy array of coefficients)
    f = -np.array(score)

    # Set of decision variables indices
    set_I = range(1, len(f) + 1)

    # Add binary decision variables to the model
    x_vars = {i: plp.LpVariable(f"x_{i}", cat=plp.LpBinary) for i in set_I}

    # Constraints indices
    set_J = range(1, len(RHS) + 1)

    # Coefficients for <= constraints
    a = {(j, i): LHS[j - 1, i - 1] for j in set_J for i in set_I}
    b = {j: RHS[j - 1] for j in set_J}

    # Coefficients for == constraints
    eLHS_shape = eLHS.shape
    c = {(j, i): eLHS[j - 1, i - 1] for j in range(1, eLHS_shape[0] + 1) for i in set_I}
    d = {j: eRHS[j - 1] for j in range(1, eLHS_shape[0] + 1)}

    # Adding <= constraints
    for j in set_J:
        opt_model += (plp.lpSum(a[j, i] * x_vars[i] for i in set_I) <= b[j], f"constraint1_{j}")

    # Adding == constraints
    for j in range(1, eLHS_shape[0] + 1):
        opt_model += (plp.lpSum(c[j, i] * x_vars[i] for i in set_I) == d[j], f"constraint2_{j}")

    # Objective function coefficients
    f = {i: -score[i - 1] for i in set_I}

    # Objective function
    opt_model += plp.lpSum(x_vars[i] * f[i] for i in set_I)

    t_start = time.perf_counter()

    # Solve the model
    opt_model.solve()

    t_stop = time.perf_counter()

    # Assign solution to a DataFrame
    opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
    opt_df.reset_index(inplace=True)
    opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: plp.value(item))

    # Output model details
    print(f"Optimization time: {t_stop - t_start} seconds")
    print(f"Optimization status: {plp.LpStatus[opt_model.status]}")

    return opt_df["solution_value"].to_numpy()
