from datetime import datetime

import folium
import numpy as np
import pandas as pd
import seaborn as sns
from skyfield.sgp4lib import EarthSatellite
from skyfield.api import wgs84
from tqdm import tqdm


def plotRequests(df: pd.DataFrame, fname: str, **kwargs):
    """
    Plot requests as marker in folium map
    """

    # Initialize the map
    m = folium.Map(location=[20, 0], zoom_start=2)

    fg_request_markers = folium.FeatureGroup(name="Requests Area")
    fg_request_names = folium.FeatureGroup(name="Requests IDs", show=False)

    clr_map = sns.color_palette(kwargs.get("color_palette", "flare"), n_colors=df["strips"].max()).as_hex()

    for i in range(0, df.shape[0]):  # For each row in the data
        # Popup message on click
        popup_text = f"""
        <table>
        <caption>({df.iloc[i]['wgs84'].latitude.degrees}, {df.iloc[i]['wgs84'].longitude.degrees})</caption>
            <tr>
                <td style="font-weight: bold;">ID:</td>
                <td style="text-align: center;">{df.iloc[i]['id']}</td>
            </tr>
            <tr style="background-color: var(--bs-gray-300);">
                <td style="font-weight: bold;">Area Covered:</td>
                <td style="text-align: center;">{df.iloc[i]['area']:.3f}km<sup>2</sup></td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Stereo Request?:</td>
                <td style="text-align: center;">{df.iloc[i]['stereo']}</td>
            </tr>
            <tr style="background-color: var(--bs-gray-300);">
                <td style="font-weight: bold;">Required Strips:</td>
                <td style="text-align: center;">{df.iloc[i]['strips']}</td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Duration:</td>
                <td style="text-align: center;">{df.iloc[i]['duration']:.3f}</td>
            </tr>
            <tr style="background-color: var(--bs-gray-300);">
                <td style="font-weight: bold;">Priority:</td>
                <td style="text-align: center;">{df.iloc[i]['priority']}</td>
            </tr>
            <tr">
                <td style="font-weight: bold;">Price:</td>
                <td style="text-align: center;">{df.iloc[i]['price']}</td>
            </tr>
        </table>
        """
        # (Lat, Lon): ({df.iloc[i]['lat']}, {df.iloc[i]['lon']})<br>
        popup = folium.Popup(popup_text, lazy=True)

        # Requested area marker
        width = np.sqrt(df.iloc[i]["area"])  # Side length of area - Assumed square
        latlon_add = (width / 2) / 111.03  # Convert km to degrees - Assumed spherical earth
        lat, lon = df.iloc[i]["wgs84"].latitude.degrees, df.iloc[i]["wgs84"].longitude.degrees
        bbox = [
            [lat + latlon_add, lon - latlon_add],
            [lat + latlon_add, lon + latlon_add],
            [lat - latlon_add, lon + latlon_add],
            [lat - latlon_add, lon - latlon_add],
            [lat + latlon_add, lon - latlon_add],
        ]  # Bounding box of the area

        clr_line = clr_map[df.iloc[i]["strips"] - 1] if df.iloc[i]["stereo"] == False else "black"
        fg_request_markers.add_child(folium.PolyLine(bbox, popup=popup, color=clr_line, weight=8))
        fg_request_names.add_child(
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    icon_size=(100, 30),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 20pt">{df.iloc[i]["id"]}</div>',
                ),
            )
        )
    m.add_child(fg_request_markers)
    m.add_child(fg_request_names)

    fname = f"{fname}.html" if fname[-5:] != ".html" else fname

    # folium.LayerControl().add_to(m)
    m.save(fname)

    return m


def plotSatPaths(
    m: folium.Map,
    df: pd.DataFrame,
    sat_data: list[EarthSatellite,],
    t_acq: list[datetime,],  # Acquisition times,
    fname: str,
    **kwargs,
):
    fg = folium.FeatureGroup(name="Satellite Paths")

    clr_map = sns.color_palette(kwargs.get("color_palette", "crest"), n_colors=len(sat_data)).as_hex()

    for i, sat in enumerate(tqdm(sat_data, desc="Plotting Satellite Paths")):
        ts = sat.epoch.ts
        for j, (t, pos) in enumerate(zip(t_acq, sat.at(ts.from_datetimes(t_acq)))):
            mask = (df["acq_time"] == t.strftime("%Y-%m-%d %H:%M:%S")) & (df["sat_name"] == sat.name)
            acq_possible = list(df[mask]["ID"])
            popup_text = f"""
            <table>
            <caption>{sat.name}</caption>
                <tr>
                    <td style="font-weight: bold;">Time:</td>
                    <td style="text-align: center;">{t}</td>
                </tr>
                <tr style="background-color: var(--bs-gray-300);">
                    <td style="font-weight: bold;">Possible Acquisitions:</td>
                    <td style="text-align: center;">{acq_possible}</td>
                </tr>
            </table>
            """
            lat, lon = wgs84.latlon_of(pos)

            fg.add_child(folium.CircleMarker(location=(lat.degrees, lon.degrees), radius=4, opacity=1 - (j / (1.3 * len(t_acq))), popup=folium.Popup(popup_text), color=clr_map[i]))

    m.add_child(fg)

    fname = f"{fname}.html" if fname[-5:] != ".html" else fname

    folium.LayerControl().add_to(m)
    m.save(fname)

    return m


def plotSolutions(req_feasible: pd.DataFrame, schedules: np.ndarray, m: folium.Map, **kwargs):

    if m is None:
        raise Exception('To plot solution, you need to generate the map alongside the scenario. Add the kwarg "plot_request_map" to the "scenario" function to do so.')

    idx = np.where(schedules)[0]
    tasks = req_feasible.iloc[idx][["sat_latlon", "req_latlon"]].values.tolist()
    folium.PolyLine(tasks, color=kwargs.get("clr", "black"), weight=2.5, opacity=1).add_to(m)
    m.save(kwargs.get("output", "eospython_solution.html"))

    print("Plotted!")
