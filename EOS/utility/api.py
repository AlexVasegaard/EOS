import time
from functools import lru_cache
from datetime import datetime, timezone
import requests
from tqdm import trange


@lru_cache
def getWeatherForecast(lat: float, lon: float, api_key: str, units: str = "standard", mode: str = "json", cnt: int = 0, lang: str = "en") -> dict:
    """https://openweathermap.org/forecast5

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        api_key (str): Your unique API key (you can always find it on your account page under the "API key" tab).
        units (str, optional): Units of measurement. `standard`, `metric` and `imperial` units are available. Defaults to `standard`.
        mode (str, optional): Response format. Defaults to `json`.
        cnt (int, optional): Number of days returned. Defaults to `40`.
        lang (str, optional): You can use this parameter to get the output in your language. Defaults to `en`.

    Returns:
        data (dict): Cloud data.
    """

    while True:
        call_str = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units={units}&mode={mode}&cnt={cnt}&lang={lang}"

        try:
            response = requests.get(call_str)
            if response.status_code != 200:
                raise Exception("API Error")
            response = response.json()

        except Exception as e:  # I haven't had the time to determine the exact response that is returned when the API is overloaded.
            print(f"WARNING: OpenWeatherMap API might be overloaded. Trying again in a few seconds... ({e})")
            for _ in trange(10):
                time.sleep(1)
            continue

        return {datetime.fromtimestamp(i["dt"], tz=timezone.utc): i["clouds"]["all"] for i in response["list"]}


@lru_cache
def getWeatherHistory(lat: float, lon: float, api_key: str, date_start: str, date_end: str, tz: str = "local", lang: str = "en", units: str = "M") -> dict:
    """https://www.weatherbit.io/api/historical-weather-hourly

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        api_key (str): Your unique API key.
        start_date (str): `YYYY-MM-DD` or `YYYY-MM-DD:HH`
        end_date (str): `YYYY-MM-DD` or `YYYY-MM-DD:HH`
        tz (str, optional): `utc` or `local`. Defaults to `local` (`start_date` and `end_date` is given in local time).
        lang (str, optional): Language. Defaults to `en` (English).
        units (str, optional): Units. Defaults to `M` (Metric).

    Returns:
        response (dict): API Response.

    API Response Example:
    ```
    {
             "timezone":"America\/New_York",
             "state_code":"NC",
             "lat":35.7721,
             "lon":-78.63861,
             "country_code":"US",
             "station_id":"723060-13722",
             "sources":["723060-13722", "USC00445050", "USW00013732"],
             "data":[
                {
                   "rh":32,
                   "wind_spd":6.7,
                   "wind_gust_spd": 9.4,
                   "slp":1020.3,
                   "h_angle":15,
                   "azimuth":25,
                   "dewpt":-7.5,
                   "snow":0,
                   "uv":0,
                   "wind_dir":220,
                   "weather":{
                      "icon":"c01n",
                      "code":"800",
                      "description":"Clear sky"
                   },
                   "pod":"n",
                   "vis":1.5,
                   "precip":0,
                   "elev_angle":-33,
                   "ts":1483232400,
                   "pres":1004.7,
                   "datetime":"2018-05-01:06",
                   "timestamp_utc":"2015-05-01T06:00:00",
                   "timestamp_local":"2015-05-01T02:00:00",
                   "revision_status":"final",
                   "temp":8.3,
                   "dhi":15,
                   "dni":240.23,
                   "ghi":450.9,
                   "solar_rad":445.85,
                   "clouds":0
                }, ...
             ],
             "city_name":"Raleigh",
             "city_id":"4487042"
          }
    ```
    """
    call_str = f"https://api.weatherbit.io/v2.0/history/hourly?lat={lat}&lon={lon}&start_date={date_start}&end_date={date_end}&key={api_key}&tz={tz}&lang={lang}&units={units}"
    response = requests.get(call_str).json()
    return response


@lru_cache
def getTLE(ID: int):
    """Get TLE data for a satellite.

    Args:
        ID (int): NORAD ID of the satellite.
    """

    if ID == 38755:
        tle = [
            "SPOT 6                  ",
            "1 38755U 12047A   24221.89458683  .00001184  00000+0  26457-3 0  9994",
            "2 38755  98.2193 287.9766 0000998  95.5328 264.5985 14.58552851634326",
            "",
        ]
    elif ID == 40053:
        tle = [
            "SPOT 7                  ",
            "1 40053U 14034A   24221.91905855  .00001527  00000+0  33138-3 0  9992",
            "2 40053  98.1550 287.8182 0001379 107.6366 252.4985 14.59500687538283",
            "",
        ]
    else:
        tle = requests.get(f"https://www.celestrak.com/NORAD/elements/gp.php?CATNR={ID}&FORMAT=TLE", verify=True, timeout=20).text.split("\r\n")

    return {"name": tle[0], "line1": tle[1], "line2": tle[2]}


if __name__ == "__main__":
    import os
    import json
    from pprint import pprint

    # Example Usage
    lat, lon = 57.0467, 9.935
    api_key = os.getenv("OWM_API_KEY")
    out = getWeatherForecast(lat, lon, api_key, cnt=2)
    pprint(out)

    api_key = os.getenv("WB_API_KEY")
    date_start = "2024-08-01:01"
    date_end = "2024-08-03:10"
    out = getWeatherHistory(lat, lon, api_key, date_start=date_start, date_end=date_end, tz="utc")
    with open("weather_history.json", "w") as f:
        json.dump(out, f, indent=4)
    pprint(out)
