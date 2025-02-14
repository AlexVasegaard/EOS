import gurobipy as grb
import numpy as np
import pandas as pd
from cvxopt import matrix


def solve(score, RHS, LHS, eRHS, eLHS, **kwargs):

    # Initialize model
    opt_model = grb.Model(name="BLP_Model")

    # Objective coefficients (assuming score is a list or numpy array of coefficients)
    f = -np.array(score)

    # Set of decision variables indices
    set_I = range(1, len(f) + 1)

    # Add binary decision variables to the model
    x_vars = {i: opt_model.addVar(vtype=grb.GRB.BINARY, name=f"x_{i}") for i in set_I}

    # Constraints indices
    set_J = range(1, len(RHS) + 1)

    # Coefficients for <= constraints
    a = {(j, i): np.array(matrix(LHS))[j - 1, i - 1] for j in set_J for i in set_I}
    b = {(j): RHS[j - 1] for j in set_J}

    # Coefficients for == constraints
    c = {(j, i): eLHS[j - 1, i - 1] for j in range(1, eLHS.shape[0]) for i in set_I}
    d = {(j): eRHS[j - 1] for j in range(1, eLHS.shape[0])}

    # Adding <= constraints
    for j in set_J:
        opt_model.addConstr(lhs=grb.quicksum(a[j, i] * x_vars[i] for i in set_I), sense=grb.GRB.LESS_EQUAL, rhs=b[j], name=f"constraint1_{j}")

    # Adding == constraints
    for j in range(1, eLHS.shape[0]):
        opt_model.addConstr(lhs=grb.quicksum(c[j, i] * x_vars[i] for i in set_I), sense=grb.GRB.EQUAL, rhs=d[j], name=f"constraint2_{j}")

    # Objective function coefficients
    f = {i: -score[i - 1] for i in set_I}

    # Objective function
    objective = grb.quicksum(x_vars[i] * f[i] for i in set_I)
    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)

    # Optimize the model
    opt_model.optimize()

    # Assign solution to a DataFrame
    opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
    opt_df.reset_index(inplace=True)
    opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.X)

    print(f"Optimization time: {opt_model.Runtime}")
    print(f"Optimization status: {opt_model.Status}")

    return opt_df["solution_value"].to_numpy()
