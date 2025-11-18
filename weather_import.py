import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def get_hourly_weather(latitude, longitude, start_date, end_date, timezone="GMT", hourly_variables=None):
    """
    Fetches hourly weather data from Open-Meteo and returns it as a Pandas DataFrame.

    Parameters:
    - latitude (float): Latitude of the location.
    - longitude (float): Longitude of the location.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - timezone (str): Timezone for the data (default: "GMT").
    - hourly_variables (list): List of hourly variables to fetch. 
                               Defaults to ["temperature_2m", "precipitation", "rain", "snowfall", "surface_pressure", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "is_day", "direct_radiation_instant", "diffuse_radiation_instant", "direct_normal_irradiance_instant", "global_tilted_irradiance_instant", "terrestrial_radiation_instant", "shortwave_radiation_instant", "cloud_cover"].

    Returns:
    - pd.DataFrame: A DataFrame containing the hourly weather data.
    """
    
    # 1. Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # 2. Define API parameters
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    if hourly_variables is None:
        hourly_variables = ["temperature_2m", "precipitation", "rain", "snowfall", "surface_pressure", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "is_day", "direct_radiation_instant", "diffuse_radiation_instant", "direct_normal_irradiance_instant", "global_tilted_irradiance_instant", "terrestrial_radiation_instant", "shortwave_radiation_instant", "cloud_cover"]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "hourly": hourly_variables
    }

    # 3. Make the API call
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0] # Process the first location
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame() # Return empty DF on failure

    # 4. Process hourly data
    hourly = response.Hourly()
    
    # Collect the data into a dictionary
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    # Dynamically add requested variables to the dictionary
    for i, variable in enumerate(hourly_variables):
        # openmeteo_requests returns a flat float array for each variable
        hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

    # 5. Create and return DataFrame
    df = pd.DataFrame(data=hourly_data)
    
    # Optional: Filter the exact date range requested (API sometimes returns slight buffer)
    # df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    return df

# --- Usage Example ---
if __name__ == "__main__":
    # Example inputs
    lat = 52.52
    lon = 13.41
    start = "2023-01-01"
    end = "2023-01-31"
    

    print("Fetching weather data...")
    df_weather = get_hourly_weather(lat, lon, start, end)

    if not df_weather.empty:
        print("\nData fetched successfully:")
        print(df_weather.head())
        print(f"\nTotal rows: {len(df_weather)}")
    else:
        print("Failed to fetch data.")

