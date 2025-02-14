import os
import time
from math import ceil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from skyfield.api import load, wgs84
from skyfield.framelib import itrs
from skyfield.sgp4lib import EarthSatellite
from tqdm import tqdm
from utility.api import getWeatherForecast, getWeatherHistory

OWM_API_KEY = os.getenv("OWM_API_KEY")
WB_API_KEY = os.getenv("WB_API_KEY")

PLANETS = load("de421.bsp")
SUN, EARTH = PLANETS["sun"], PLANETS["earth"]

rng = np.random.default_rng(
    seed=42,
)


def genCloudCover(
    lat: float,
    lon: float,
    parameter: float,
    alpha: float = None,
    beta: float = None,
) -> float:
    """Generate cloud cover based on latitude, longitude, and other parameters.
    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        parameter (float): Additional parameter to adjust cloud cover.
        alpha (float, optional): Alpha parameter for cloud cover calculation. Defaults to None.
        beta (float, optional): Beta parameter for cloud cover calculation. Defaults to None.

    Returns:
        float: Generated cloud cover value.
    """
    lat1 = np.cos(((lat + 90) / 180) * 20)  # to mimic higher around equator and one upper and lower quantile
    lon1 = np.cos(((lon + 180) / 360) * 50)
    lat2 = np.cos(((lat + 90) / 180) * 50)
    lon2 = np.cos(((lon + 180) / 360) * 30)
    lat3 = np.cos(((lat + 90) / 180) * 100)
    lon3 = np.cos(((lon + 180) / 360) * 80)

    if alpha is None or beta is None:
        cloud = (((3 * lat1 * lon1 + 1.5 * lat1 + 2 * lat2 * lon2 + 1 * lat3 * lon3) * 100 / 8 + 70) * 2 - 150) / 1.5  # it now has a range from -50 to 150
    else:  # alpha and beta set
        cloud = 110 - (((lat1 * lon1 + lat2 * lon2 + 2) / (alpha)) - (beta - alpha) / beta) * 110  # to mimic scewedness!

    cloud_stoc1 = cloud + parameter * rng.integers(-10, 10)  # to generate bias in both ends
    cloud_stoc2 = max(min(cloud_stoc1 + rng.integers(-5, 5), 100), 0)

    return cloud_stoc2


def genIDs(number_of_requests: int) -> list[str,]:
    """Generate a list of IDs for a given number of requests.
    Args:
        number_of_requests (int): The total number of requests.
    Returns:
        list[str]: A list of IDs, where each ID is a zero-padded string.
    Example:
    ```
    idCreate(100)
    >>> ["001", "002", ... , "009", "010", "011", ... , "099", "100"]
    ```
    """

    padding = len(str(number_of_requests))
    return [str(i).zfill(padding) for i in range(1, number_of_requests + 1)]


def genCustomerRequests(
    req_n: int,
    **kwargs,
) -> pd.DataFrame:
    """Generates an assortment of customer requests.

    Args:
        req_n (int): Number of requests to generate.

    Kargs:
        data_location (str): Path to the Excel file containing city locations.
        area_range (tuple): Range of the area (float) of the requests in square kilometers. Drawn uniformly. Default is `(1, 1000)`.
        swarth_width (float): Swarth width (float) of the satellite in kilometers. Default is `60`.
        duration_range (tuple): Range of the duration (float) of the requests. Drawn uniformly. Default is `(2, 8)`.
        price_range (tuple): Range of the price (int) of the requests. Drawn discrete uniformly. Default is `(500, 1500)`.
        req_types (dict): Dictionary of request types (int) and their probabilities (float). Default is `{1: 0.05, 2: 0.2, 3: 0.50, 4: 0.25}`.

    Returns:
        pd.DataFrame: Customer request database.
    """

    print("Creating Customer Database...")
    t_start = time.perf_counter()

    # Create a unique ID for each request
    req_ID = genIDs(req_n)

    city_locations = pd.read_excel(kwargs.get("data_location", "EOS/data/city_locations.xlsx"), index_col=0).sample(
        req_n, random_state=rng
    )  # Latitude and longitude of the requests are sampled from Excel file with coordinates for different cities around the world.

    # The area which the image must cover
    low, high = kwargs.get("area_range", (1, 1000))
    req_area = rng.uniform(low, high, req_n)  # km^2

    # Determine if an image request is a stereo request based on a Poisson distribution. req_stereo[i] == 1 => Stereo request
    req_stereo = rng.poisson(lam=0.1, size=req_n) > 0

    # Number of strips required to cover the image area
    req_strips = (
        2.25 * np.sqrt(req_area) / kwargs.get("swarth_width", 60)
    )  # 2.25 is magical number... it increases the possibility of strips acquisitions when area distribution is small
    req_strips = np.astype(np.ceil(req_strips), int)
    # Force single strip acquisition for stereo requests
    req_strips[req_stereo > 0] = 1

    # Duration of the request
    low, high = kwargs.get("duration_range", (2, 8))
    req_duration = rng.uniform(low, high, size=req_n)

    # Monetary cost of the request
    low, high = kwargs.get("price_range", (500, 1500))
    req_price = rng.integers(low, high, size=req_n)

    req_types = kwargs.get("req_types", {1: 0.05, 2: 0.2, 3: 0.50, 4: 0.25})

    req_pri = rng.choice(list(req_types.keys()), size=req_n, replace=True, p=list(req_types.values()))

    req_lats = city_locations["lat"].to_numpy()
    req_lons = city_locations["lng"].to_numpy()
    # req_heights = np.linalg.norm(
    #     np.array([wgs84.latlon(lat, lon).itrs_xyz.km for lat, lon in zip(req_lats, req_lons)]),
    #     ord=2,
    #     axis=1,
    # )

    df = pd.DataFrame(
        {
            "id": req_ID,  # Request ID
            "wgs84": [wgs84.latlon(lat, lon) for lat, lon in zip(req_lats, req_lons)],  # Request position using WGS84
            "area": req_area,  # Request Area Coverage
            "stereo": req_stereo,  # Stereo bool
            "strips": req_strips,  # Number of Strips Required to Cover Area
            "duration": req_duration,  # Duration of Request
            "priority": req_pri,  # Request Priority
            "price": req_price,  # Request Monetary Price
            "age": rng.integers(1, 14, size=req_n),  # Days since request submission
        }
    )

    print(f"Customer Database Created! ({timedelta(seconds=time.perf_counter() - t_start)})")

    return df


def genFeasibleRequests(
    sat_data: list[EarthSatellite,],
    t_acq: list[datetime,],  # Acquisition times
    db: pd.DataFrame,
    angle_matrix: np.ndarray,
    **kwargs,
) -> pd.DataFrame:
    """_summary_

    Args:
        sat_data (list[EarthSatellite,]): _description_
        t_acq (list[datetime,]): _description_
        angle_matrix (np.ndarray): _description_

    Kwargs:
        use_cloud_cover_data (bool, optional): Fetches actual cloud cover data using an API call to openweathermap. (Not yet implemented!)
        max_cloud_cover (float, optional): Changes how the cloud cover generation works, allowing removal of execisive cloud cover. Defaults to 50.

    Raises:
        Exception: _description_

    Returns:
        pd.DataFrame: _description_
    """

    data = {
        "ID": [],
        "stereo": [],
        "sat_name": [],
        "sat_latlon": [],
        "sat_xyz": [],
        "req_latlon": [],
        "req_xyz": [],
        "acq_time": [],
        "area": [],
        "strips": [],
        "duration": [],
        "angle": [],
        "priority": [],
        "price": [],
        "uncertainty": [],
        "age": [],
        "sun_elevation": [],
    }

    sat_locs = []
    for earthSat in sat_data:
        ts = earthSat.epoch.ts
        pos = earthSat.at(ts.from_datetimes(t_acq))
        sat_locs.append(pos)  # Satellite locations at each time step

    req_feasible = np.argwhere(~np.isnan(angle_matrix.values))

    # TODO: Speed this up
    # Sun Altitude over Local Horizon
    sun_alt = []
    pbar = tqdm(total=len(sat_data) * len(db), desc="Calculating Sun Elevations")
    ts = None
    for earthSat in sat_data:
        if ts == earthSat.epoch.ts:  # Speedup in case all satellites utilize the same time scale
            sun_alt.append(sun_alt[-1])
            pbar.update(len(db))
            continue
        ts = earthSat.epoch.ts  # Needed if the satellites utilizes different time scales
        arr = []
        t = ts.from_datetimes(t_acq)
        for geoPos in db["wgs84"]:
            loc = EARTH + geoPos
            alt, *_ = loc.at(t).observe(SUN).apparent().altaz()
            arr.append(alt.degrees)
            pbar.update(1)
        sun_alt.append(arr)
    pbar.close()

    for idx_sat, idx_t, idx_req in tqdm(req_feasible, desc="Listing Feasible Requests"):
        alt = sun_alt[idx_sat][idx_req][idx_t]  # Sun Altitude over Local Horizon
        if alt < 0:
            continue
        data["sun_elevation"].append(alt)  # Sun Altitude over Local Horizon
        db_entry = db.iloc[idx_req]
        data["ID"].append(db_entry["id"])

        data["sat_name"].append(sat_data[idx_sat].name)  # Satellite name
        sat_pos = wgs84.geographic_position_of(sat_locs[idx_sat][idx_t])
        data["sat_latlon"].append((sat_pos.latitude.degrees, sat_pos.longitude.degrees))  # Satellite location at time of acquisition
        data["sat_xyz"].append(sat_locs[idx_sat][idx_t].frame_xyz(itrs).km)  # ITRS coordinates of the satellite at time of acquisition

        data["req_latlon"].append((db_entry["wgs84"].latitude.degrees, db_entry["wgs84"].longitude.degrees))  # Request location
        data["req_xyz"].append(db_entry["wgs84"].itrs_xyz.km)  # ITRS coordinates of the request location

        data["acq_time"].append(t_acq[idx_t])  # Acquisition time (.strftime("%Y-%m-%d %H:%M:%S")
        data["area"].append(db_entry["area"])  # Request area
        data["strips"].append(db_entry["strips"])  # Number of strips
        data["duration"].append(db_entry["duration"])  # Request duration
        # data["distance"].append(np.linalg.norm(sat_locs[idx_sat][idx_t] - db_entry["ITRS"].itrs_xyz.km, ord=2))  # L2 distance between satellite and request
        data["angle"].append(angle_matrix.values[idx_sat, idx_t, idx_req])  # Off-nadir angle

        data["uncertainty"].append(idx_t / len(t_acq))  # Uncertainty as a function of the number of acquisition times

        data["stereo"].append(db_entry["stereo"])  # Stereo request
        data["priority"].append(db_entry["priority"])  # Request priority
        data["price"].append(db_entry["price"])  # Request price
        data["age"].append(db_entry["age"])  # Days since request submission

    df = pd.DataFrame(data)

    unique_locs = np.unique(df["req_latlon"])
    if kwargs.get("use_cloud_cover_data"):  # FIXME # Use cloud cover data from the API
        raise Exception("Cloud cover data from the API is not implemented yet.")
        # TODO: Only use "forecast" if current time is such that we need to predict the weather. Otherwise use "historic".

        # Split acquisition times into historic and forecast
        mask = df["acq_time"].dt.date < datetime.now().date()
        acq_historic = df.loc[mask]
        acq_forecast = df.loc[~mask]

        # Get cloud cover data for future acquisitions
        timespan_forcast_hours = (acq_forecast["acq_time"].max() - acq_forecast["acq_time"].min()).seconds // 3600
        cnt = ceil((timespan_forcast_hours + 1) / 3)  # Timestamps requested by the API (3 hour resolution for OWM)
        unique_times = np.unique(acq_forecast["acq_time"])
        for idx_row, group in tqdm(acq_forecast.groupby(["req_latlon"]), desc="Getting Cloud Cover Forecast"):
            lat, lon = group["req_latlon"].iloc[0]
            clouds = getWeatherForecast(lat, lon, OWM_API_KEY, cnt=cnt)
            time_map = {ut: min(clouds.keys(), key=lambda x: abs((x - ut).total_seconds())) for ut in unique_times}
            df.loc[group.index, "cc_est"] = df.loc[group.index, "acq_time"].map(lambda x: clouds[time_map[x]], na_action="ignore")

        for idx_row, group in tqdm(acq_historic.groupby(["req_latlon"]), desc="Getting Cloud Cover History"):
            pass
            # TODO: Implement historic data
        # if mode == "historic":
        #     # for each unique requests - collect data
        #     t_from = int(str(time_slots[0])[11:13])
        #     t_to = t_from + int(np.ceil(horizon_dt)) + 1
        #     date_start = str(time_slots[0])[0:10]
        #     date_end = str(time_slots[0] + datetime.timedelta(days=1))[0:10]
        #     cc_locs = []
        #     for i, (lat, lon) in enumerate(unique_locs):
        #         response_dict = getWeatherHistory(lat, lon, WB_API_KEY, date_start, date_end)
        #         weather_list = [response_dict["data"][j]["clouds"] for j in range(t_from, t_to)]
        #         cc_locs.append(weather_list)
        #         print(f"{i} out of {len(unique_locs)}: {weather_list}")
        #         time.sleep(1)  # to avoid overloading the API | ?? calls per minute for free account

        #     # for all requests - allocate information
        #     unique_locs_list = np.concatenate(unique_locs).reshape((len(unique_locs), 2))
        #     for i in range(0, len(df)):
        #         which_h = np.where(np.arange(t_from, t_to) == int(str(df["time"][i])[11:13]))[0][0]
        #         which_unique_loc = np.where(np.sum(unique_locs_list == df["request location"][i], axis=1) == 2)[0][0]

        #         cc_est = cc_locs[which_unique_loc][which_h]
        #         # we assume the observed weather is somewhat near the forecast
        #         df["cloud cover estimate"][i] = cc_est
        #         df["cloud cover real"][i] = max(
        #             min(
        #                 cc_est + (n / len(location_slots[0])) * random.randint(-20, 20),
        #                 100,
        #             ),
        #             0,
        #         )
    else:  # Generate some pseudo cloud cover data
        similarity_parameter = rng.uniform(0, 1)
        cc = {(lat, lon): genCloudCover(lat, lon, similarity_parameter) for (lat, lon) in unique_locs}

        df["cc_est"] = df["req_latlon"].map(lambda x: max(min(cc[x] + rng.integers(-5, 5), 100), 0))  # Pseudo estimated cloud cover
        df["cc_real"] = df["cc_est"] + df["uncertainty"] * rng.integers(-10, 10, size=len(df))
        df["cc_real"] = df["cc_real"].map(lambda x: max(min(x, 100), 0))  # Pseudo real cloud cover

    # Remove bad weather attempts
    df = df.drop(df.loc[df["cc_real"] > kwargs.get("max_cloud_cover", 50)].index)

    # Sort by satellite and time, and reset the index
    df = df.sort_values(by=["sat_name", "acq_time"]).reset_index(drop=True)

    return df
