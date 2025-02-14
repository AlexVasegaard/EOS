import numpy as np
import pandas as pd
from tqdm import tqdm


def topsis(data: pd.DataFrame, criteria_max: list[str,], criteria_weights: dict[str, float]) -> np.ndarray:
    """
    The Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) is a multi-criteria decision analysis method. TOPSIS argues that the chosen alternative should have the shortest euclidian distance from the positive ideal solution and the longest euclidian distance from the negative ideal solution.

    https://en.wikipedia.org/wiki/TOPSIS

    Args:
        data (pd.DataFrame): Dataframe containing the alternatives and their corresponding parameters/information.
        criteria_max (list[str,]): Whuch columns (criteria) are to be maximized instead of minimized.
        criteria_weights (dict[str, float]): The weight of each column (criteria) in the scoring process.

    Returns:
        np.ndarray: The score (relative closeness to the worst solution) of each alternative.
    """

    data /= (data**2).sum(axis="index") ** 0.5  # Normalize the data

    criteria_weights = pd.Series(criteria_weights) / sum(criteria_weights.values())  # Normalize weights
    data *= criteria_weights  # Weight the data

    criteria_min = [col for col in data.columns if col not in criteria_max]  # Criteria to be minimized
    ideal_criteria = pd.concat(
        [
            pd.concat([data.loc[:, criteria_max].max(axis="rows"), data.loc[:, criteria_max].min(axis="rows")], axis="columns"),
            pd.concat([data.loc[:, criteria_min].min(axis="rows"), data.loc[:, criteria_min].max(axis="rows")], axis="columns"),
        ],
        axis="rows",
    )  # Ideal best and worst solutions for each criteria
    ideal_criteria.columns = ["best", "worst"]

    dist_best = np.linalg.norm(data - ideal_criteria["best"], ord=2, axis=1)  # Distance to best solution
    dist_worst = np.linalg.norm(data - ideal_criteria["worst"], ord=2, axis=1)  # Distance to worst solution

    scores = dist_worst / (dist_best + dist_worst)  # Score is the relative closeness to the worst solution

    return scores


def electre_III(data: pd.DataFrame, criteria_thresholds: pd.DataFrame, criteria_max: list[str,], criteria_weights: dict[str, float]) -> np.ndarray:
    """Elimination and Choice Expressing Reality (ELECTRE) is a family of multi-criteria decision analysis methods that originated in Europe in the mid-1960s. The methods aim to rank alternatives or determine the best alternative from a set of alternatives, each of which is characterized by multiple criteria. The methods are based on the concept of outranking, which means that an alternative a is preferred to another alternative b if a is at least as good as b in all criteria and is better than b in at least one criterion.

    https://en.wikipedia.org/wiki/%C3%89LECTRE

    Args:
        data (pd.DataFrame): Dataframe containing the alternatives and their corresponding parameters/information.
        criteria_thresholds (pd.DataFrame): The Preference (p), Indifference (q), and Veto (v) thresholds.
        criteria_max (list[str]): Which columns (criteria) are to be maximized instead of minimized.
        criteria_weights (dict[str, float]): The weight of each column (criteria) in the scoring process.

    Returns:
        np.ndarray: The average score of each alternative.
    """

    # data = data.copy()  # Make sure we don't operate on the actual dataframe

    # We can't just multiply by -1, as it would have an unintentional impact on the weighted average at the end
    data.loc[:, criteria_max] = data[criteria_max].max(axis=0) - data[criteria_max]

    criteria_weights = pd.Series(criteria_weights) / sum(criteria_weights.values())  # Normalize weights
    data, criteria_thresholds = data.align(criteria_thresholds, axis=1)  # Align threshold columns with data columns

    p, q, v = criteria_thresholds.loc["p"], criteria_thresholds.loc["q"], criteria_thresholds.loc["v"]  # Preference (p), Indifference (q), and Veto (v) thresholds

    scores = np.empty(shape=(len(data), len(data)))  # Holds all the scores between alternatives
    for idx, alt in tqdm(data.iterrows(), desc="ELECTRE", total=len(data)):
        diff = data - alt

        mask = (q < diff) & (diff < p)
        phi = (p - diff[mask]) / (p - q)  # Concordance Index
        phi[diff <= q] = 1
        phi.fillna(0, inplace=True)

        mask = (p < diff) & (diff < v)
        d = (diff[mask] - p) / (v - p)  # Discordance Index
        d[diff >= v] = 1
        d = d.fillna(0)

        sigma = (phi * criteria_weights).sum(axis=1)  # Weighted Credibility Index
        all_true = d.le(sigma, axis="index").all(axis="columns")
        sigma[~all_true] *= ((1 - d[~all_true]) / (1 - sigma[~all_true])).prod(axis=1)
        scores[idx] = sigma.to_list()

    return np.mean(scores, axis=1)
