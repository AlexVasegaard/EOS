import datetime
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric
from skyfield.sgp4lib import EarthSatellite

rng = np.random.default_rng(seed=42)


def xarrayVectorNorm(x: xr.DataArray, dim) -> xr.DataArray:
    return xr.apply_ufunc(np.linalg.norm, x, input_core_dims=[[dim]], kwargs=dict(ord=2, axis=-1))


def getFeasibleAngles(
    sat_data: list[EarthSatellite,],
    t_acq: list[datetime,],  # Acquisition times
    db: pd.DataFrame,
    ona_max: float,  # Maximum Off Nadir Angle, to ensure quality of image in degrees (https://www.euspaceimaging.com/what-is-ona-off-nadir-angle-in-satellite-imagery/)
    **kwargs,
) -> np.ndarray:
    """_summary_

    Args:
        sat_data (list[EarthSatellite,]): _description_
        t_acq (list[datetime,]): Acquisition times
        ona_max (float): Maximum Off Nadir Angle, to ensure quality of image in degrees (https://www.euspaceimaging.com/what-is-ona-off-nadir-angle-in-satellite-imagery/)

    Kwargs:
        ona_min (float, optional): Minimum Off Nadir Angle in degrees, to ensure quality of image in degrees (https://www.euspaceimaging.com/what-is-ona-off-nadir-angle-in-satellite-imagery/).

    Returns:
        np.ndarray: _description_
    """

    print(f"Calculating Off-Nadir Angles for Request-Satelite Pairs...")

    t_start = time.perf_counter()
    # # Here we want to compare the angle between the satellite and the request location. If the angle is larger than ona_max, the request cannot be fullfilled.

    # Calculate satellite locations at every time step

    # Ensure we use the correct time-scale for each satellite
    sat_locs = []
    for sat in sat_data:
        ts = sat.epoch.ts
        sat_locs.append(sat.at(ts.from_datetimes(t_acq)).frame_xyz(itrs).km.T)  # Satellite locations at each time step. Note that we use ITRS coordinates

    # First we get the satellite position vectors. These are drawn from the center of the earth to the satellite.
    sat_pos = xr.DataArray(
        sat_locs, dims=("sat", "t", "pos"), coords={"sat": [sat.name for sat in sat_data], "t": [t.strftime("%Y-%m-%d %H:%M:%S") for t in t_acq]}
    )  # sat_pos[i, j] -> [x,y,z]

    # Next we get the ITRS coordinates of the request locations
    req_pos = xr.DataArray(db["wgs84"].apply(lambda x: x.itrs_xyz.km).to_list(), dims=("id", "pos"), coords={"id": db["id"]})  # req_pos[i] -> [x,y,z]

    # Now we calculate the directional vectors from the request to the satellite locations
    dir_vec = req_pos - sat_pos

    # We normalize the vectors beforehand, allowing us to skip calculating magnitudes later
    req_pos_norm = req_pos / xarrayVectorNorm(req_pos, dim="pos")  # Unit vector
    sat_pos_norm = sat_pos / xarrayVectorNorm(sat_pos, dim="pos")  # Unit vector
    dir_vec_norm = dir_vec / xarrayVectorNorm(dir_vec, dim="pos")  # Unit vector

    # We calculate the entry-wise dot product (Projection Matrix)
    proj_nadir = (-sat_pos_norm * dir_vec_norm).sum(dim="pos")  # -sat_pos_norm make vector point in the right direction
    proj_visible = (-dir_vec_norm * req_pos_norm).sum(dim="pos")  # -dir_vec_norm make vector point in the right direction

    ### DEBUG ###
    if False:
        print("\n### DEBUG ###")
        # idx_sat, idx_t, idx_req = 1, 357, 8
        # print(f"Angle between Request Position {idx_req} and Satellite Position {idx_t} for Satellite {idx_sat}: {np.degrees(np.acos(proj_nadir.isel(sat=idx_sat, t=idx_t, id=idx_req))):.2f}º")
        str_sat, str_t, str_req = "SPOT 6", "2024-07-11 10:45:40", "0316"  # idx_sat, idx_t, idx_req = 0, 197, 315
        print(
            f'Angle between Request {str_req} and Satellite Position at {str_t} for Satellite "{str_sat}": {np.degrees(np.acos(proj_nadir.sel(sat=str_sat, t=str_t, id=str_req))):.2f}º'
        )
        print(f"Pruned @ {ona_max}º? {proj_nadir.sel(sat=str_sat, t=str_t, id=str_req) < np.cos(np.radians(ona_max))}")

        resolution = 100
        x, y, z = 0, 0, 0
        r = 6371  # Earth's radius
        u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
        X = r * np.cos(u) * np.sin(v) + x
        Y = r * np.sin(u) * np.sin(v) + y
        Z = r * np.cos(v) + z

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    opacity=0.2,
                    showscale=False,
                    text="Earth",
                    hoverinfo="skip",
                ),
                go.Scatter3d(
                    x=[sat_pos.sel(sat=str_sat, t=str_t)[0]],
                    y=[sat_pos.sel(sat=str_sat, t=str_t)[1]],
                    z=[sat_pos.sel(sat=str_sat, t=str_t)[2]],
                    mode="markers",
                    # text=["Satellite Position"],
                    name=f"Satellite Position ({str_sat} | {str_t})",
                ),
                go.Scatter3d(
                    x=[req_pos.sel(id=str_req)[0]],
                    y=[req_pos.sel(id=str_req)[1]],
                    z=[req_pos.sel(id=str_req)[2]],
                    mode="markers",
                    # text=["Request Position"],
                    name=f"Request Position ({str_req})",
                ),
                go.Scatter3d(
                    x=[0, sat_pos.sel(sat=str_sat, t=str_t)[0]],
                    y=[0, sat_pos.sel(sat=str_sat, t=str_t)[1]],
                    z=[0, sat_pos.sel(sat=str_sat, t=str_t)[2]],
                    mode="lines",
                    line=dict(color="darkblue", width=4),
                    # text=["Nadir Vector"],
                    name="Nadir Vector",
                ),
                go.Scatter3d(
                    x=[sat_pos.sel(sat=str_sat, t=str_t)[0], sat_pos.sel(sat=str_sat, t=str_t)[0] + dir_vec.sel(sat=str_sat, t=str_t, id=str_req)[0]],
                    y=[sat_pos.sel(sat=str_sat, t=str_t)[1], sat_pos.sel(sat=str_sat, t=str_t)[1] + dir_vec.sel(sat=str_sat, t=str_t, id=str_req)[1]],
                    z=[sat_pos.sel(sat=str_sat, t=str_t)[2], sat_pos.sel(sat=str_sat, t=str_t)[2] + dir_vec.sel(sat=str_sat, t=str_t, id=str_req)[2]],
                    mode="lines",
                    line=dict(
                        color="darkred",
                        width=4,
                        # dash="dot",
                    ),
                    # text=["Direction Vector"],
                    name="Direction Vector",
                ),
            ]
        )

        fig.write_html("angle_plot.html")
        print("### ##### ###\n")
    ### ##### ###

    # At this point we could take the arccos of proj to get the angle between the vectors, but we can reduce the computations by simply comparing in the cos-domain instead
    proj_nadir = proj_nadir.where(proj_nadir > np.cos(np.radians(ona_max)), other=np.nan, drop=False)  # Angle will never be above 180º, so we can compare in the cos-domain
    proj_nadir = proj_nadir.where(proj_visible > 0, other=np.nan, drop=False)  # Ensure the request is visible (max angle < 90º)

    if kwargs.get("ona_min"):
        proj_nadir = proj_nadir.where(proj_nadir < np.cos(np.radians(ona_max)), other=np.nan, drop=False)

    proj_nadir = np.degrees(np.acos(proj_nadir))

    # Thus angle_matrix[i,j,k] is the angle between request position j and satellite position k for satellite i

    print(f"Angle Calculations Completed! ({timedelta(seconds=time.perf_counter() - t_start)})")
    print(f"Reachable Requests: {1-np.isnan(proj_nadir).sum()/proj_nadir.size:.1%} (#{int((~np.isnan(proj_nadir)).sum())})\n")

    return proj_nadir
