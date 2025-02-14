import time
from datetime import datetime, timedelta, timezone
from math import floor

import numpy as np
import pandas as pd
from dataGenerator import genCustomerRequests, genFeasibleRequests
from LPP import createLPPFormulation
from skyfield.api import EarthSatellite, load
from tqdm import tqdm
from utility.api import getTLE
from utility.plotting import plotRequests, plotSatPaths, plotSolutions
from utility.utils import getFeasibleAngles

rng = np.random.default_rng(seed=42)


def scenario(
    db: pd.DataFrame,
    NORAD_ids: list,
    horizon_t0: str,  # "YYYY-MM-DD HH:MM:SS"
    n_requests: int = 1000,
    dt_acq: int = 20,  # Seconds
    horizon_dt: int = 8,  # Hours
    ona_max: float = 30,  # degrees
    angular_velocity: float = 2.5,  # degrees per second
    cam_resolution: float = 1,  # m^2 per pixel
    **kwargs,
) -> dict:
    """Constructs a scenario for the Multi-Satellite Image Acquisition Scheduling Problem.

    Args:
        db (pd.DataFrame): Customer database. Will automatically generate on if none is provided to allow for testing.
        NORAD_ids (list): Satellite Catalog Numbers specifying which satellites to utilize. (https://en.wikipedia.org/wiki/Satellite_Catalog_Number)
        horizon_t0 (str): When to start calculating the satellite paths. Must be "YYYY-MM-DD HH:MM:SS".
        n_requests (int): The number of requests to generate. Only utilized when no customer database is provided.
        dt_acq (int, optional): Time between image attempts in seconds, i.e. discretization of the satellite path. Defaults to 20.
        horizon_dt (int, optional): Time to schedule into the future in hourse. Defaults to 8.
        ona_max (float, optional): Maximum Off Nadir Angle in degrees, to ensure quality of image in degrees (https://www.euspaceimaging.com/what-is-ona-off-nadir-angle-in-satellite-imagery/). Defaults to 30.
        angular_velocity (float, optional): The agility of the satellite in degrees per second. Defaults to 2.5.
        cam_resolution (float, optional): Image resolution in m2 per pixel. Defaults to 1.

    Kwargs:
        timezone (datetime.timezone, optional). Timezone for positional calculations. Defaults to "UTC".
        sat_data (list[skyfield.sgp4lib.EarthSatellite,], optional). Allows the custom satellites to be specified using the SkyField library. Overwrites the automatic construction from the NORADS IDs.
        filepath_requests_map (str, optional). Where to save the map of the requests. Defaults to "requests_map".
        filepath_satellite_map (str, optional). Where to save the map of the satellite locations. Defaults to "satellite_map".
        See
        - genCustomerRequest,
        - getFeasbileAngles,
        - genFeasibleRequests,
        - plotRequests,
        - plotSatPaths

        for further parameters.


    Returns:
        Dictionary of
            "LPP": The LPP formulation. See createLPPFormulation for keys.
            "df": The customer database.
            "m": The folium map.
            "sats": The NORAD IDs.
    """
    t_start = time.perf_counter()
    print("## Starting Scenario Generation Process... ##\n")

    if db is None:
        db = genCustomerRequests(n_requests, **kwargs)

    # Horizon start time
    horizon_t0 = datetime.strptime(horizon_t0, "%Y-%m-%d %H:%M:%S").replace(tzinfo=kwargs.get("timezone", timezone.utc))
    print(f"{horizon_t0} to {horizon_t0 + timedelta(hours=horizon_dt)}")

    # Fetch TLE data
    ts = load.timescale()

    # Create the satellite objects
    sat_data = kwargs.get("sat_data", [EarthSatellite(**getTLE(id), ts=ts) for id in tqdm(NORAD_ids, desc="Fetching TLEs")])

    n_acq_points = floor((horizon_dt * 60 * 60) / dt_acq)  # Points in the horizon
    t_acq = [horizon_t0 + timedelta(seconds=i * dt_acq) for i in range(n_acq_points)]  # Discretize horizon

    ona_feasible = getFeasibleAngles(sat_data, t_acq, db, ona_max, **kwargs)

    req_feasible = genFeasibleRequests(sat_data, t_acq, db, angle_matrix=ona_feasible, **kwargs)

    m = None
    if kwargs.get("plot_request_map"):
        m = plotRequests(db, fname=kwargs.get("filepath_requests_map", "requests_map"), **kwargs)

    # plot satellite path
    if kwargs.get("plot_satellite_map"):
        if m is None:
            m = plotRequests(db, fname=kwargs.get("filepath_requests_map", "requests_map"), **kwargs)
            # m = folium.Map(location=[20, 0], zoom_start=2)

        plotSatPaths(
            m,
            req_feasible,
            sat_data,
            t_acq,
            fname=kwargs.get("filepath_satellite_map", "satellite_map"),
            **kwargs,
        )

    LPP = createLPPFormulation(req_feasible, t_acq, cam_resolution=cam_resolution, angular_velocity=angular_velocity)

    print(f"Scenario generation completed in {time.perf_counter() - t_start:.2f}s")

    return {
        "LPP": LPP,
        "df": db,
        "m": m,
        "sats": NORAD_ids,
    }


def solve(
    LPP,
    scoring_method: str = "TOPSIS",  # "TOPSIS", "ELECTRE", or "WSA"
    solution_method: str = "PuLP",  # "gurobi", "PuLP", "cplex", "VNS", "random", or "DAG"
    **kwargs,
):

    t_start = time.perf_counter()

    t_start_scoring = time.perf_counter()
    print(f'Scoring using "{scoring_method}"...')
    # IDENTIFY CRITERIA TO INCLUDE IN SCORING PROCEDURE
    criteria_cols = kwargs.get("criteria_cols", ["area", "angle", "sun_elevation", "cc_est", "priority", "price", "age", "uncertainty"])
    criteria_weights = kwargs.get("criteria_weights", {col: 1 for col in criteria_cols})
    criteria_max = kwargs.get("criteria_max", ["area", "sun_elevation", "price", "age"])

    data = LPP["req_feasible"][criteria_cols]  #  We only want to base the score on the specifed criteria

    if scoring_method == "TOPSIS":
        from scoring import topsis
        score = topsis(data, criteria_max, criteria_weights)

    elif scoring_method == "ELECTRE":
        from scoring import electre_III
        criteria_thresholds = kwargs.get("criteria_thresholds", None)
        if criteria_thresholds is None:
            raise Exception("Criteria thresholds must be provided for ELECTRE method!")
        score = electre_III(data, criteria_thresholds, criteria_max, criteria_weights)

    elif scoring_method == "WSA":  # Weight Space Analysis https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10436690&tag=1
        raise Exception("WSA not implemented yet!")

    LPP["req_feasible"]["score"] = score  # Add the score to the dataframe
    print(f"Scoring Completed in {timedelta(seconds=time.perf_counter() - t_start_scoring)}s")

    t_start_solving = time.perf_counter()
    print(f'Solving using "{solution_method}"...')
    match solution_method.lower():
        case "gurobi":
            from solvers import solver_gurobi
            x = solver_gurobi.solve(score=score, **LPP)
        case "pulp":
            from solvers import solver_pulp
            x = solver_pulp.solve(score=score, **LPP)
        case "cplex":
            from solvers import solver_cplex
            x = solver_cplex.solve(score=score, **LPP)
        case "glpk":
            from solvers import solver_glpk
            x = solver_glpk.solve(score=score, bin_vars=range(len(LPP["req_feasible"])), **LPP)
        case "dag":
            raise Exception('"DAG" solver not yet implemented!')
            from solvers import solver_dag
            x = solver_dag.solve(
                score=score,
                edge_search_depth=kwargs.get("edge_search_depth", 25),  # Only used for solution_method="DAG"
                **LPP,
            )
        case "vns":
            from solvers import solver_vns
            x = solver_vns.solve(score, LPP, LPP["req_feasible"], max_time=120, max_iter_N=1000)
        case "random":
            raise Exception('"random" solver not yet implemented!')
        case _:
            raise Exception(f'Solution method "{solution_method}" not recognized! Please use "gurobi", "PuLP", "cplex", "GLPK", "DAG", "VNS", or "RANDOM"')

    print(f"Solution found in {timedelta(seconds=time.perf_counter() - t_start_solving)}s")

    print(f"Total Time: {timedelta(seconds=time.perf_counter() - t_start)}s")

    return {"solution": np.squeeze(x), "score": score}


def evaluate(x_data, x_res, **kwargs):
    # scenario generation specific evalation metrics
    reqss_m = int(len(np.unique(x_data.pf_df["ID"])))
    atts_m = int(len(x_data.pf_df))
    constraintss_m = int(x_data.LPP.eRHS.shape[0] + x_data.LPP.RHS.shape[0])
    angles_m = np.mean(x_data.pf_df["angle"])
    areas_m = np.mean(x_data.pf_df["area"])
    prices_m = np.mean(x_data.pf_df["price"])
    sun_elevations_m = np.mean(x_data.pf_df["sun elevation"])
    ccs_m = np.mean(x_data.pf_df["cloud cover real"])
    prio_m = np.mean(x_data.pf_df["priority"])

    # solution specific metrics
    acq = int(np.sum(x_res.x))
    profit = np.sum(x_data.pf_df["price"].iloc[np.where(x_res.x)])
    avg_cloud = np.mean(x_data.pf_df["cloud cover real"].iloc[np.where(x_res.x)])
    cloud_good = int(np.sum(x_data.pf_df["cloud cover real"].iloc[np.where(x_res.x)] < 10))
    cloud_bad = int(np.sum(x_data.pf_df["cloud cover real"].iloc[np.where(x_res.x)] > 30))
    avg_angle = np.mean(x_data.pf_df["angle"].iloc[np.where(x_res.x)])
    angle_good = int(np.sum(x_data.pf_df["angle"].iloc[np.where(x_res.x)] <= 10))
    angle_bad = int(np.sum(x_data.pf_df["angle"].iloc[np.where(x_res.x)] >= 30))
    avg_priority = np.mean(x_data.pf_df["priority"].iloc[np.where(x_res.x)])
    priority1 = int(np.sum(x_data.pf_df["priority"].iloc[np.where(x_res.x)] == 1))
    priority2 = int(np.sum(x_data.pf_df["priority"].iloc[np.where(x_res.x)] == 2))
    priority3 = int(np.sum(x_data.pf_df["priority"].iloc[np.where(x_res.x)] == 3))
    priority4 = int(np.sum(x_data.pf_df["priority"].iloc[np.where(x_res.x)] == 4))
    sunelevation = np.mean(x_data.pf_df["sun elevation"].iloc[np.where(x_res.x)])
    totalarea = np.sum(x_data.pf_df["area"].iloc[np.where(x_res.x)])

    evaluate.scenario = pd.DataFrame(
        {
            "metric": [
                "requests",
                "attempts",
                "constraints",
                "avg angle",
                "avg area",
                "avg price",
                "avg sun elevation",
                "avg cloud cover",
                "avg priority",
            ],
            "value": [
                reqss_m,
                atts_m,
                constraintss_m,
                angles_m,
                areas_m,
                prices_m,
                sun_elevations_m,
                ccs_m,
                prio_m,
            ],
        }
    )

    evaluate.solution = pd.DataFrame(
        {
            "metric": [
                "acquisitions",
                "total profit",
                "avg cloud cover",
                "cloud cover < 10",
                "cloud cover > 30",
                "avg angle",
                "angle < 10",
                "angle > 30",
                "avg priority",
                "priority 1",
                "priority 2",
                "priority 3",
                "priority 4",
                "avg sun elevation",
                "total area",
            ],
            "value": [
                acq,
                profit,
                avg_cloud,
                cloud_good,
                cloud_bad,
                avg_angle,
                angle_good,
                angle_bad,
                avg_priority,
                priority1,
                priority2,
                priority3,
                priority4,
                sunelevation,
                totalarea,
            ],
        }
    )

    return evaluate


if __name__ == "__main__":
    from pprint import pprint

    params = dict(
        db=None,
        NORAD_ids=[38755, 40053],  # Spot 6 and 7 - Assumed to be heterogenous an capable of acquiring the customer requests in the database,
        horizon_t0="2024-08-29 20:40:00",
        n_requests=1000,
        simplify=True,
        dt_acq=20,
        horizon_dt=8,  # Planning horizon in hours
        ona_max=30,
        angular_velocity=30 / 12,
        cam_resolution=1,
        capacity_limit=1000000,
        use_cloud_cover_data=False,
        plot_request_map=True,
        plot_satellite_map=True,
    )
    print("## Scenario Params ##")
    pprint(params)
    print("#" * len("## Scenario Params ##") + "\n")
    s = scenario(**params)

    print("## Solution Params ##")
    params = dict(
        scoring_method="TOPSIS",
        solution_method="PULP",
        criteria_thresholds=pd.DataFrame(
            {
                "area": {"q": 0, "p": 50, "v": 1000},
                "angle": {"q": 2, "p": 5, "v": 40},
                "sun_elevation": {"q": 2, "p": 5, "v": 40},
                "cc_est": {"q": 0, "p": 5, "v": 15},
                "price": {"q": 0, "p": 1000, "v": 10000},
                "priority": {"q": 0, "p": 1, "v": 2},
                "age": {"q": 0, "p": 1, "v": 5},
                "uncertainty": {"q": 0, "p": 2, "v": 5},
            }
        ),
    )
    pprint(params)
    print("#" * len("## Solution Params ##") + "\n")
    solution = solve(s["LPP"], **params)

    plotSolutions(req_feasible=s["LPP"]["req_feasible"], schedules=solution["solution"], m=s["m"])

    print("Finished!")
