import numpy as np


def getFeasbilityMatrix(req_feasible):
    strips_perf_df = np.where((req_feasible["stereo"] < 2)[req_feasible["strips"] >= 2])[0]
    strips_IDs = list(np.unique(req_feasible["ID"][strips_perf_df]))
    number_of_strips_IDs = list()
    Strips = list()
    for i in range(0, len(strips_IDs)):
        range_strip = list(np.where(req_feasible["ID"] == strips_IDs[i])[0])
        strips_for_i = req_feasible["strips"][range_strip[0]]
        if len(range_strip) < strips_for_i:
            continue
        for a in range(0, int(len(range_strip) - strips_for_i)):
            Strips.append(range_strip[a:])
            number_of_strips_IDs.append(strips_for_i)

    Strips_constraint = np.zeros((len(Strips), len(req_feasible)))
    for i in range(len(Strips)):
        Strips_constraint[i, Strips[i][0]] = number_of_strips_IDs[i]
        Strips_constraint[i, Strips[i][1:]] = -1

    return Strips_constraint
