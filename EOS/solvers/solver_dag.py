import numpy as np
import time
from datetime import timedelta
import networkx as nx
import rustworkx as rx

# TODO: Redo this module!


def pathRemove(path, I, edges, stereo, strips, strips_num_acq, score):
    path[I] = 0
    if np.any(path) == False:
        return path
    else:
        number_before_I = int(np.sum(path[0:I]))
        if number_before_I > 0:
            from_a = np.where(path)[0][number_before_I - 1]  # the number before variable starts from 1
        else:
            from_a = 0
        number_after_I = int(np.sum(path[I:]))
        if number_after_I > 0:
            from_b = np.where(path)[0][number_before_I]
        else:
            return path

        search_range = [from_a, from_b]

        into = np.where(edges[search_range[0], :])[0]
        feasible_nodes = into[np.where(edges[into, search_range[1]])[0]]
        feasible_nodes = np.setdiff1d(feasible_nodes, I)
        if len(feasible_nodes) == 0:
            return path
        else:
            longest_path_tempp = [[] for ii in range(0, len(feasible_nodes))]
            weight_of_path_tempp = np.zeros((len(feasible_nodes)))
            for iii in range(0, len(feasible_nodes)):
                path_temp = np.copy(path)
                path_temp[feasible_nodes[iii]] = 1
                if all(strips @ path_temp <= strips_num_acq) and all(stereo @ path_temp == 0):
                    longest_path_tempp[iii] = path_temp
                    weight_of_path_tempp[iii] = score[feasible_nodes[iii]]
            if np.max(weight_of_path_tempp) == 0:
                return path
            else:
                return longest_path_tempp[np.argmax(weight_of_path_tempp)]


def pathInsert(x, I, keep, edges, stereo, strips, strips_num_acq, score):
    x[I] = 1
    if np.any(strips @ x > strips_num_acq):
        # remove least contributing interdep node
        interdep = np.where(strips[np.where(strips[:, I])[0], :])[1]
        interdep_i_acq = np.intersect1d(interdep, np.where(x)[0])
        if keep != []:
            interdep_i_acq = np.setdiff1d(interdep_i_acq, keep)
        interdep = np.delete(interdep_i_acq, np.where(interdep_i_acq == I)[0])  # single
        x = pathRemove(x, interdep[np.argmin(score[interdep])], edges, stereo, strips, strips_num_acq, score)

    if np.any(stereo @ x != 0):
        stereo_idx = np.where(stereo[np.where(stereo @ x != 0)[0], :] == 1)[1]  # single
        x = pathRemove(x, stereo_idx[np.argmin(score[stereo_idx])], edges, stereo, strips, strips_num_acq, score)

    r1 = np.squeeze(edges[np.where(x[0:I])[0], I])
    r2 = np.squeeze(edges[I, np.where(x[I:])[0] + I])
    feasible_maneuver = np.concatenate((r1.reshape((r1.size)), r2.reshape((r2.size))))
    if np.sum(feasible_maneuver == False) > 1:  # because I is infeasible with it self
        infeas = np.where(x)[0][np.where((np.where(x)[0] != I) * 1 + ~feasible_maneuver.astype(bool) == 2)[0]]

        if len(infeas) < 2:
            x = pathRemove(x, infeas[0], edges, stereo, strips, strips_num_acq, score)
        else:
            infeas_lims = infeas[[0, len(infeas) - 1]]
            x[infeas] = 0
            for infs in infeas_lims:
                x = pathRemove(x, infs, edges, stereo, strips, strips_num_acq, score)

    return x


def solve(score, edge_search_depth, req_feasible, stereo, F, **kwargs):
    """https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10436690&tag=1"""

    t_start = time.perf_counter()

    # Create edges matrix (no loops)
    edges0 = F == 0
    edges = np.triu(edges0, 1)  # remove loops and directed (cannot go backwards)

    # Algorithm should not include interdependent sets, that is, attempts representing the same request

    # stereo and strips interdependencies

    # strips

    all_strip_reqs_id = req_feasible["ID"].iloc[np.where(req_feasible["strips"] >= 1)]
    unique_strip_reqs = np.unique(all_strip_reqs_id)
    strips = np.zeros((len(unique_strip_reqs), len(req_feasible)))
    strips_num_acq = list()
    for ii in range(0, len(unique_strip_reqs)):
        index_i = all_strip_reqs_id.iloc[np.where(all_strip_reqs_id == unique_strip_reqs[ii])[0]].index
        strips[ii, index_i] = 1
        strips_num_acq.append(max(req_feasible[["stereo", "strips"]].iloc[index_i[0]] + np.array([1, 0])))

    ##EXTENDED LONGEST PATH ALGORITHM
    longest_path_to_node = [[] for _ in range(0, len(edges))]
    weight_of_path = np.zeros((len(edges)))

    for i in range(0, len(edges)):
        incomming_neighbours = list(np.where(edges[0:i, i])[0])
        # if zero longest path is just it se lf.
        if len(incomming_neighbours) == 0:
            if all(stereo[:, i] == 0):  # non-stereo
                longest_path_to_node[i].append(i)
                weight_of_path[i] = score[i]
            else:
                weight_of_path[i] = 0
        else:
            ##initiate loop to find largest path not including an interdependent node
            sort_neighbour = np.argsort(weight_of_path[incomming_neighbours])[::-1]
            vertice_which = np.array(incomming_neighbours)[sort_neighbour]
            # naive shorting of incoming neighbours - parameter to how deep it should investigate
            depth = min(edge_search_depth, len(incomming_neighbours))
            longest_path_temp = [[] for _ in range(0, depth)]
            weight_of_path_temp = np.zeros((depth))
            for j in range(0, depth):
                path = np.concatenate((longest_path_to_node[vertice_which[j]], [i]))
                x = np.zeros((len(edges)))
                x[path.astype(int)] = 1
                if all(strips @ x <= strips_num_acq):
                    if all(stereo @ x == 0):
                        if j == 0:
                            longest_path_to_node[i] = np.where(x)[0]
                            weight_of_path[i] = score @ x
                            break
                        else:
                            longest_path_temp[j] = x
                            weight_of_path_temp[j] = score @ x
                    else:
                        # ADD stereo or path is not possible
                        stereo_set = np.where(stereo[np.where(stereo[:, i] == -1)[0], :] == 1)[1]
                        if len(stereo_set) == 0:
                            weight_of_path_temp[j] = 0
                            continue
                        else:
                            x_stereo = pathInsert(x, stereo_set[0], i, edges, stereo, strips, strips_num_acq, score)
                            longest_path_temp[j] = x_stereo
                            weight_of_path_temp[j] = score @ x_stereo
                else:
                    # remove least contributing interdep node
                    if all(stereo[:, i] == 0):  # non-stereo
                        np.intersect1d(np.where(strips[np.where(strips @ x > strips_num_acq)[0], :])[1], np.where(x[:i])[0])
                        interdep_i = np.where(strips[np.where(strips[:, i])[0], :])[1]
                        interdep_i_acq = np.intersect1d(interdep_i, np.where(x)[0])
                        interdep_i_acq = np.setdiff1d(interdep_i_acq, i)

                        min_interdep_i = np.argmin(score[interdep_i_acq])
                        min_ilegal_node = interdep_i_acq[min_interdep_i]
                        longest_path_temp[j] = pathRemove(x, min_ilegal_node, edges, stereo, strips, strips_num_acq, score)
                    else:
                        ind_stereo = np.where(stereo[np.where(stereo[:, i] == -1)[0], :i] == 1)[1]
                        if len(ind_stereo) == 0:
                            # legal path to this one is not feasible
                            weight_of_path_temp[j] = 0
                            continue
                        else:
                            # we now know that another set of stereo attempts are performed of the same stereo request as node i is trying to acquire
                            # so that pair has to be terminated and the not-included should be included:
                            stereo_pairs = np.intersect1d(ind_stereo, path)
                            if len(stereo_pairs) == 0:  # the other pair is not included
                                int_stereo_set = np.intersect1d(np.where(strips[np.where(strips @ x > strips_num_acq)[0], :])[1], np.where(x[:i])[0])
                                x[int_stereo_set] = 0  # investigated after adding the others
                                ##
                                if len(ind_stereo) > 1:  # mistake if this can happen
                                    print(ind_stereo)
                                ##

                                new_stereo = int(ind_stereo[0])
                                x = pathInsert(x, new_stereo, i, edges, stereo, strips, strips_num_acq, score)
                                for i_s in int_stereo_set:
                                    x = pathRemove(x, i_s, edges, stereo, strips, strips_num_acq, score)  # investigates the first of the prior removed stereo attempt
                                longest_path_temp[j] = x
                            else:
                                min_stereo = stereo_pairs[np.argmin(score[stereo_pairs])]  # modified
                                longest_path_temp[j] = pathRemove(x, min_stereo, edges, stereo, strips, strips_num_acq, score)
                    weight_of_path_temp[j] = longest_path_temp[j] @ score

            if len(longest_path_to_node[i]) == 0:
                longest_path_to_node[i] = np.where(longest_path_temp[np.argmax(weight_of_path_temp)])[0]
                weight_of_path[i] = np.max(weight_of_path_temp)

    x = np.zeros((len(edges)))
    x[longest_path_to_node[np.argmax(weight_of_path)]] = 1

    print(f"Optimization time: {timedelta(seconds=time.perf_counter() - t_start)}s")

    return x
