#solarAirQuality.py

def get_air_quality_data_dynamic(latitude, longitude, start_date, end_date):
    from openmeteo_requests import Client
    import requests_cache
    from retry_requests import retry
    import pandas as pd

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ["pm10", "pm2_5", "uv_index", "dust", "aerosol_optical_depth"]
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "pm10": hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
            "uv_index": hourly.Variables(2).ValuesAsNumpy(),
            "dust": hourly.Variables(3).ValuesAsNumpy(),
            "aerosol_optical_depth": hourly.Variables(4).ValuesAsNumpy(),
        }
        return pd.DataFrame(data=hourly_data)
    except Exception as e:
        print("Error getting air quality data:", e)
        return pd.DataFrame()


