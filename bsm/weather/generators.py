"""
Defines classes for generating weather data for the thermal simulation.
"""
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from bsm.weather.api_client import get_hourly_weather
from bsm.weather.sky_temperature import SkyTemperatureCalculator
from bsm.solar.position import SolarPositionCalculator
from bsm.weather.ground_temperature import KusudaGroundTemperature

class SimpleSinusoidal:
    """
    Generates a simple sinusoidal weather profile based on config parameters.
    """
    def __init__(self, config):
        """
        Initializes the weather generator.
        
        Args:
            config (dict): The 'weather' section of the main config dictionary.
        """
        self.config = config

    def generate_weather_data(self, time_hours):
        """
        Generates the time-series list of weather data.

        Args:
            time_hours (np.array): A numpy array of the simulation time in hours
                                   for each step.
        
        Returns:
            list[dict]: A list of weather data dictionaries, one for each time step.
        """
        weather_data = []
        
        # Pre-calculate config values
        solar_max = self.config['solar_max_irradiance_w_m2']
        temp_base = self.config['temp_base_c']
        temp_amp = self.config['temp_amplitude_c']
        temp_phase = self.config['temp_phase_shift_hr']

        # Fix: Iterate directly over values, not enumerate tuple (index, value)
        for t_hr in time_hours:
            # --- Simple Sinusoidal Weather Model ---
            
            # Calculate solar radiation, ensuring it's non-negative
            solar_rad = max(0, solar_max * np.cos(2 * np.pi * (t_hr - 12) / 24))
            
            # Calculate air temperature based on a sinusoidal model
            air_temp = (temp_base + 
                        temp_amp * np.cos(2 * np.pi * (t_hr - temp_phase) / 24))
            
            weather_data.append({
                'air_temp_c': air_temp,
                'wind_speed_local_ms': 3.0, # Example fixed value
                'wind_speed_10m_ms': 2.5,   # Example fixed value
                'wind_direction_deg': 180,  # Example fixed value
                'solar_irradiance_w_m2': solar_rad
            })
            
        return weather_data


class EPWWeatherLoader:
    """
    Loads EnergyPlus Weather (EPW) files for BESTEST validation.

    EPW files are standard EnergyPlus weather files with 8 header lines
    followed by 8760 hourly data records (for a full year).
    """
    def __init__(self, epw_path: str):
        """
        Initializes the EPW weather loader.

        Args:
            epw_path (str): Path to the EPW file
        """
        self.epw_path = epw_path
        self.df_hourly = None
        self.sky_temp_calc = SkyTemperatureCalculator(model='berdahl_martin')

        # Parse location from EPW header for solar position calculations
        self._parse_location()

        # Initialize solar position calculator with location
        self.solar_calc = SolarPositionCalculator(
            latitude=self.latitude,
            longitude=self.longitude,
            timezone_offset_hours=self.tz_offset
        )

        self._parse_epw()

    def _parse_location(self):
        """
        Parse location data from EPW file header (line 1).

        EPW LOCATION line format:
        LOCATION,City,State,Country,Source,WMO,Lat,Lon,TZ,Elev
        Example: LOCATION,DENVER INTL AP,CO,USA,TMY3,725650,39.83,-104.65,-7.0,1650.0
        """
        try:
            with open(self.epw_path, 'r') as f:
                location_line = f.readline().strip()

            parts = location_line.split(',')
            if parts[0] != 'LOCATION':
                raise ValueError(f"Expected LOCATION header, got: {parts[0]}")

            self.location_name = f"{parts[1]}, {parts[2]}, {parts[3]}"
            self.latitude = float(parts[6])
            self.longitude = float(parts[7])
            self.tz_offset = float(parts[8])
            self.elevation = float(parts[9])

            print(f"EPW Location: {self.location_name}")
            print(f"  Latitude: {self.latitude}°, Longitude: {self.longitude}°")
            print(f"  Timezone: UTC{self.tz_offset:+.1f}, Elevation: {self.elevation}m")

        except Exception as e:
            raise RuntimeError(f"Error parsing EPW location header: {e}")

    def _parse_epw(self):
        """
        Parses the EPW file and stores hourly weather data.

        EPW Format:
        - Lines 1-8: Header information
        - Lines 9+: Hourly data (comma-separated)
        """
        try:
            # Read EPW file, skipping first 8 header lines
            df = pd.read_csv(
                self.epw_path,
                skiprows=8,
                header=None,
                low_memory=False
            )

            # EPW column definitions (0-based pandas indexing, EPW Field N = df[N-1])
            # Reference: https://bigladdersoftware.com/epx/docs/8-3/auxiliary-programs/
            #            energyplus-weather-file-epw-data-dictionary.html
            #
            # EPW/TMY3 format - using standard EnergyPlus field definitions:
            #
            # df[0-5]: Year, Month, Day, Hour (1-24), Minute, Data Source/Uncertainty
            # df[6]: Dry Bulb Temperature (°C) - EPW Field 7
            # df[13]: Global Horizontal Radiation GHI (Wh/m²) - EPW Field 14 ← USED FOR SOLAR
            # df[14]: Direct Normal Radiation DNI (Wh/m²) - EPW Field 15 ← FOR FUTURE USE
            # df[15]: Diffuse Horizontal Radiation DHI (Wh/m²) - EPW Field 16 ← FOR FUTURE USE
            # df[20]: Wind Direction (degrees) - EPW Field 21
            # df[21]: Wind Speed (m/s) - EPW Field 22
            #
            # Note: Field 11 (df[10]) may contain extraterrestrial radiation values
            # Note: Field 13 (df[12]) contains non-zero values at night - quality flags
            # Note: Wh/m² values are numerically equivalent to W/m² for hourly averages

            # Create datetime index
            df['datetime'] = pd.to_datetime(
                df[[0, 1, 2]].rename(columns={0: 'year', 1: 'month', 2: 'day'}),
                errors='coerce'
            ) + pd.to_timedelta(df[3] - 1, unit='h')  # Hour field is 1-24, convert to 0-23

            # Extract and rename relevant columns
            self.df_hourly = pd.DataFrame({
                'date': df['datetime'],
                'temperature_2m': df[6],  # Dry bulb temperature (°C) - EPW Field 7
                'dew_point_temp': df[7],  # Dew point temperature (°C) - EPW Field 8
                'opaque_sky_cover': df[22],  # Opaque sky cover (tenths, 0-10) - EPW Field 23
                'horizontal_ir_sky': df[12],  # Horizontal IR from sky (Wh/m²) - EPW Field 13
                'wind_speed_10m': df[21],  # Wind speed (m/s)
                'wind_direction_10m': df[20],  # Wind direction (degrees)
                'shortwave_radiation_instant': df[13],  # GHI (Wh/m²) - EPW Field 14
                'direct_normal_radiation': df[14],  # DNI (Wh/m²) - EPW Field 15
                'diffuse_horizontal_radiation': df[15]  # DHI (Wh/m²) - EPW Field 16
            })

            # Set datetime as index
            self.df_hourly.set_index('date', inplace=True)

            # === DIAGNOSTIC: Verify time indexing ===
            print("\n=== EPW TIME INDEXING CHECK ===")
            print("EPW Hour field uses 1-24 format (hour ENDING timestamp)")
            print("First 3 hours of parsed data:")
            for i in range(3):
                epw_hour = df.iloc[i][3]
                timestamp = self.df_hourly.index[i]
                ghi = self.df_hourly.iloc[i]['shortwave_radiation_instant']
                print(f"  EPW Hour {epw_hour:2.0f} → {timestamp.strftime('%Y-%m-%d %H:%M')} | GHI: {ghi:5.1f} Wh/m²")
            print("===================================\n")

            # === DIAGNOSTIC: Verify solar data columns (TEMPORARY - REMOVE AFTER VALIDATION) ===
            print("=== EPW SOLAR DATA CHECK (TMY3 FORMAT) ===")

            # Check midnight (Hour 1, index 0)
            print("Midnight check (Hour 1):")
            idx_midnight = 0
            if len(df) > idx_midnight:
                ghi_midnight = df.iloc[idx_midnight][13]  # Field 14
                dni_midnight = df.iloc[idx_midnight][14]  # Field 15
                dhi_midnight = df.iloc[idx_midnight][15]  # Field 16
                print(f"  GHI (df[13]/Field14): {ghi_midnight:.1f} Wh/m² (must be 0 at night)")
                print(f"  DNI (df[14]/Field15): {dni_midnight:.1f} Wh/m²")
                print(f"  DHI (df[15]/Field16): {dhi_midnight:.1f} Wh/m²")
                if ghi_midnight == 0 and dni_midnight == 0 and dhi_midnight == 0:
                    print("  ✓ All solar values correctly zero at midnight")
                else:
                    print(f"  ⚠️  WARNING: Non-zero solar at midnight!")

            # Check sunny morning (Hour 9, index 8)
            print("\nDaytime check (Hour 9):")
            idx_day = 8
            if len(df) > idx_day:
                ghi_day = df.iloc[idx_day][13]  # Field 14
                dni_day = df.iloc[idx_day][14]  # Field 15
                dhi_day = df.iloc[idx_day][15]  # Field 16
                field11_day = df.iloc[idx_day][10]  # Previously tested field
                print(f"  GHI (df[13]/Field14): {ghi_day:.1f} Wh/m² ← USING THIS (standard EPW)")
                print(f"  df[10]/Field11:       {field11_day:.1f} Wh/m² (extraterrestrial radiation)")
                print(f"  DNI (df[14]/Field15): {dni_day:.1f} Wh/m²")
                print(f"  DHI (df[15]/Field16): {dhi_day:.1f} Wh/m²")
                print(f"  Physics check: GHI ≈ DNI×cos(zenith) + DHI")
                print(f"  Simulation will use: {self.df_hourly.iloc[idx_day]['shortwave_radiation_instant']:.1f} Wh/m²")

            print("==========================================\n")

            # Localize to UTC for consistency with WeatherFromFile
            self.df_hourly.index = self.df_hourly.index.tz_localize('UTC')

            print(f"Successfully loaded EPW file: {self.epw_path}")
            print(f"Weather data spans: {self.df_hourly.index[0]} to {self.df_hourly.index[-1]}")

            # Calculate Kusuda-Achenbach ground temperature parameters from EPW data
            # Group by month and calculate monthly mean temperatures
            monthly_means = self.df_hourly.groupby(
                self.df_hourly.index.month
            )['temperature_2m'].mean()

            self.annual_mean_temp = monthly_means.mean()
            self.temp_amplitude = (monthly_means.max() - monthly_means.min()) / 2.0
            self.coldest_month = monthly_means.idxmin()
            self.phase_day = (self.coldest_month - 1) * 30 + 15  # Mid-month

            # Initialize ground temperature calculator
            from ground_temperature import KusudaGroundTemperature
            self.ground_temp_calc = KusudaGroundTemperature(
                annual_mean_c=self.annual_mean_temp,
                amplitude_c=self.temp_amplitude,
                phase_day=self.phase_day
            )

            print(f"\n=== Kusuda Ground Temperature Model ===")
            print(f"Annual mean: {self.annual_mean_temp:.2f}C")
            print(f"Amplitude: {self.temp_amplitude:.2f}K")
            print(f"Phase (coldest): Day {self.phase_day} (Month {self.coldest_month})")
            print(f"Ground temp range: {self.annual_mean_temp - self.temp_amplitude:.1f}C "
                  f"to {self.annual_mean_temp + self.temp_amplitude:.1f}C")
            print("========================================\n")

        except FileNotFoundError:
            raise FileNotFoundError(f"EPW file not found: {self.epw_path}")
        except Exception as e:
            raise RuntimeError(f"Error parsing EPW file {self.epw_path}: {e}")

    def generate_weather_data(self, time_hours):
        """
        Generates interpolated weather data for the simulation timesteps.

        EPW files contain exactly one year (8760 hours) of data. This method
        cycles/wraps the annual data to support warm-up periods and multi-year
        simulations.

        Args:
            time_hours (np.array): Array of simulation time in hours

        Returns:
            list[dict]: List of weather data dictionaries matching simulation format
        """
        if self.df_hourly is None:
            raise RuntimeError("EPW data has not been parsed.")

        # EPW files are annual - use modulo to wrap hours beyond 8760
        hours_per_year = 8760

        if len(self.df_hourly) != hours_per_year:
            print(f"Warning: EPW file has {len(self.df_hourly)} hours instead of expected {hours_per_year}")
            hours_per_year = len(self.df_hourly)

        weather_data = []

        for t_hr in time_hours:
            # Wrap hour index to stay within annual data
            wrapped_hour = t_hr % hours_per_year

            # Get the two bounding hours for interpolation
            hour_before = int(np.floor(wrapped_hour))
            hour_after = int(np.ceil(wrapped_hour)) % hours_per_year

            # Fractional part for interpolation
            frac = wrapped_hour - hour_before

            # Interpolate between hourly values
            # Handle wrap-around case where hour_after might equal hour_before
            if hour_after == hour_before:
                temp = self.df_hourly.iloc[hour_before]['temperature_2m']
                dew_point = self.df_hourly.iloc[hour_before]['dew_point_temp']
                opaque_sky_cover = self.df_hourly.iloc[hour_before]['opaque_sky_cover']
                horizontal_ir = self.df_hourly.iloc[hour_before]['horizontal_ir_sky']
                wind_speed = self.df_hourly.iloc[hour_before]['wind_speed_10m']
                wind_dir = self.df_hourly.iloc[hour_before]['wind_direction_10m']
                solar = self.df_hourly.iloc[hour_before]['shortwave_radiation_instant']
                dni = self.df_hourly.iloc[hour_before]['direct_normal_radiation']
                dhi = self.df_hourly.iloc[hour_before]['diffuse_horizontal_radiation']
            else:
                temp = (1 - frac) * self.df_hourly.iloc[hour_before]['temperature_2m'] + \
                       frac * self.df_hourly.iloc[hour_after]['temperature_2m']
                dew_point = (1 - frac) * self.df_hourly.iloc[hour_before]['dew_point_temp'] + \
                            frac * self.df_hourly.iloc[hour_after]['dew_point_temp']
                opaque_sky_cover = (1 - frac) * self.df_hourly.iloc[hour_before]['opaque_sky_cover'] + \
                                   frac * self.df_hourly.iloc[hour_after]['opaque_sky_cover']
                horizontal_ir = (1 - frac) * self.df_hourly.iloc[hour_before]['horizontal_ir_sky'] + \
                                frac * self.df_hourly.iloc[hour_after]['horizontal_ir_sky']
                wind_speed = (1 - frac) * self.df_hourly.iloc[hour_before]['wind_speed_10m'] + \
                             frac * self.df_hourly.iloc[hour_after]['wind_speed_10m']
                wind_dir = (1 - frac) * self.df_hourly.iloc[hour_before]['wind_direction_10m'] + \
                           frac * self.df_hourly.iloc[hour_after]['wind_direction_10m']
                solar = (1 - frac) * self.df_hourly.iloc[hour_before]['shortwave_radiation_instant'] + \
                        frac * self.df_hourly.iloc[hour_after]['shortwave_radiation_instant']
                dni = (1 - frac) * self.df_hourly.iloc[hour_before]['direct_normal_radiation'] + \
                      frac * self.df_hourly.iloc[hour_after]['direct_normal_radiation']
                dhi = (1 - frac) * self.df_hourly.iloc[hour_before]['diffuse_horizontal_radiation'] + \
                      frac * self.df_hourly.iloc[hour_after]['diffuse_horizontal_radiation']

            # Calculate sky temperature from EPW horizontal IR radiation
            # This is the EnergyPlus approach when measured/modeled IR data is available
            # Formula: IR = σ × T_sky⁴, so T_sky = (IR / σ)^0.25
            STEFAN_BOLTZMANN = 5.67e-8  # W/m²·K⁴
            ir_value = float(horizontal_ir)
            if ir_value > 0:
                # Use EPW IR data directly (more accurate than calculated)
                sky_temp_k = (ir_value / STEFAN_BOLTZMANN) ** 0.25
            else:
                # Fallback to Berdahl-Martin if IR data is missing/zero
                air_temp_k = float(temp) + 273.15
                dew_point_k = float(dew_point) + 273.15
                cloud_cover_fraction = float(opaque_sky_cover) / 10.0
                sky_temp_k = self.sky_temp_calc.calculate_sky_temperature(
                    air_temp_k=air_temp_k,
                    dew_point_temp_k=dew_point_k,
                    cloud_cover_fraction=cloud_cover_fraction
                )

            # Calculate sun position for this timestep
            # Convert simulation time to datetime
            timestamp = self.df_hourly.index[0] + timedelta(hours=float(t_hr))
            # Remove timezone for sun position calculation (uses local time)
            timestamp_local = timestamp.replace(tzinfo=None)

            zenith_deg, azimuth_deg, extraterrestrial, is_sunlit = \
                self.solar_calc.calculate_sun_position(timestamp_local)

            # Calculate ground temperature using Kusuda-Achenbach model
            # Day of year: 1-365, wrapping for multi-year simulations
            day_of_year = ((t_hr / 24.0) % 365.0) + 1.0
            ground_temp_c = self.ground_temp_calc.get_temperature(day_of_year)

            weather_data.append({
                'air_temp_c': float(temp),
                'dew_point_c': float(dew_point),
                'wind_speed_local_ms': float(wind_speed),
                'wind_speed_10m_ms': float(wind_speed),
                'wind_direction_deg': float(wind_dir),
                'solar_irradiance_w_m2': max(0.0, float(solar)),  # Legacy GHI field
                'solar_ghi_w_m2': max(0.0, float(solar)),  # Global horizontal
                'solar_dni_w_m2': max(0.0, float(dni)),  # Direct normal
                'solar_dhi_w_m2': max(0.0, float(dhi)),  # Diffuse horizontal
                'sun_zenith_deg': zenith_deg,
                'sun_azimuth_deg': azimuth_deg,
                'extraterrestrial_w_m2': extraterrestrial,
                'is_sunlit': is_sunlit,
                'sky_temp_k': sky_temp_k,
                'ground_temp_c': ground_temp_c  # Kusuda-Achenbach ground surface temperature
            })

        return weather_data


class WeatherFromFile:
    """
    Generates weather data by fetching hourly data from a file/API and interpolating
    it to the simulation time steps.

    Now provides comprehensive weather data matching EPWWeatherLoader output,
    including solar position, sky temperature, and ground temperature calculations.
    """
    def __init__(self, config):
        """
        Initializes the weather generator.

        Args:
            config (dict): The FULL main simulation config dictionary.
                           Must contain 'location' and 'simulation_settings'.
        """
        self.config = config
        self.df_hourly = None

        # Initialize location-based calculators
        loc = config.get('location', {})
        self.latitude = loc.get('latitude', 0)
        self.longitude = loc.get('longitude', 0)
        self.tz_offset = 0  # API uses UTC

        # Solar position calculator
        self.solar_calc = SolarPositionCalculator(
            latitude=self.latitude,
            longitude=self.longitude,
            timezone_offset_hours=self.tz_offset
        )

        # Sky temperature calculator (Berdahl-Martin model with humidity)
        self.sky_temp_calc = SkyTemperatureCalculator(model='berdahl_martin')

        # Ground temperature calculator (initialized after fetching data)
        self.ground_temp_calc = None

        self._fetch_hourly_data()

    def _fetch_hourly_data(self):
        """
        Fetches hourly weather data based on config location and duration.
        Also initializes the Kusuda ground temperature model.
        """
        # 1. Extract required parameters
        loc_config = self.config.get('location')
        if not loc_config:
            raise ValueError("WeatherFromFile requires a 'location' key in the config.")

        lat = loc_config.get('latitude')
        lon = loc_config.get('longitude')

        sim_settings = self.config.get('simulation_settings', {})
        start_date_str = sim_settings.get('start_date')
        duration_days = sim_settings.get('duration_days', 1)

        if not start_date_str:
            raise ValueError("WeatherFromFile requires 'start_date' in 'simulation_settings'.")

        # 2. Calculate end date
        # Note: We localize to UTC here to be consistent, though the API call uses string dates
        start_dt = pd.to_datetime(start_date_str)

        # Add 1 day buffer to ensure we cover the final hours
        end_dt = start_dt + timedelta(days=duration_days + 1)
        end_date_str = end_dt.strftime('%Y-%m-%d')

        # 3. Fetch data
        print(f"Fetching weather for Lat: {lat}, Lon: {lon} from {start_date_str} to {end_date_str}...")
        df = get_hourly_weather(lat, lon, start_date_str, end_date_str)

        if df.empty:
            raise RuntimeError("Failed to fetch weather data.")

        # 4. Process DataFrame
        # Ensure 'date' is datetime objects and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        self.df_hourly = df

        # 5. Check for soil temperature data from Open Meteo
        # If available, we can use measured values instead of Kusuda model
        self._has_soil_temp_data = all(
            col in df.columns for col in [
                'soil_temperature_0_to_7cm',
                'soil_temperature_7_to_28cm',
                'soil_temperature_28_to_100cm',
                'soil_temperature_100_to_255cm'
            ]
        )

        if self._has_soil_temp_data:
            # Check for NaN values
            soil_cols = [
                'soil_temperature_0_to_7cm',
                'soil_temperature_7_to_28cm',
                'soil_temperature_28_to_100cm',
                'soil_temperature_100_to_255cm'
            ]
            nan_count = df[soil_cols].isna().sum().sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN values in soil temperature data, will use Kusuda fallback for those")

            soil_surface = df['soil_temperature_0_to_7cm'].mean()
            soil_deep = df['soil_temperature_100_to_255cm'].mean()
            print(f"Using measured soil temperature data from Open Meteo:")
            print(f"  Surface (0-7cm): {soil_surface:.1f}°C avg, Deep (1-2.5m): {soil_deep:.1f}°C avg")
        else:
            print("Soil temperature data not available from API, using Kusuda model as fallback")

        # 6. Initialize Kusuda ground temperature model as fallback
        # Also used for interpolation at depths between measured points
        if 'temperature_2m' in df.columns and len(df) > 24:
            # Calculate approximate annual parameters from available data
            # Group by month if we have enough data, otherwise use mean and estimate amplitude
            temps = df['temperature_2m']
            mean_temp = temps.mean()

            # Estimate amplitude from diurnal range (typical annual amplitude ~ 3-4x daily amplitude)
            diurnal_range = temps.max() - temps.min()
            # For short periods, estimate annual amplitude as larger than observed
            # Typical ratio: annual amplitude ~ 10-15°C for mid-latitudes
            estimated_amplitude = max(diurnal_range / 2.0, 8.0)  # At least 8°C amplitude

            # Estimate phase based on latitude (Northern vs Southern hemisphere)
            # Northern hemisphere: coldest around day 15-45 (Jan-Feb)
            # Southern hemisphere: coldest around day 195-225 (Jul-Aug)
            if self.latitude >= 0:
                phase_day = 30  # Early February for Northern hemisphere
            else:
                phase_day = 210  # Late July for Southern hemisphere

            self.ground_temp_calc = KusudaGroundTemperature(
                annual_mean_c=mean_temp,
                amplitude_c=estimated_amplitude,
                phase_day=phase_day
            )

            if not self._has_soil_temp_data:
                print(f"Initialized Kusuda ground temperature model:")
                print(f"  Annual mean: {mean_temp:.1f}°C, Amplitude: {estimated_amplitude:.1f}K, Phase: day {phase_day}")
        else:
            # Fallback: use reasonable defaults based on latitude
            # Mid-latitude default: 10°C mean, 10°C amplitude
            default_mean = 10.0 + 0.1 * (30 - abs(self.latitude))  # Warmer near equator
            self.ground_temp_calc = KusudaGroundTemperature(
                annual_mean_c=default_mean,
                amplitude_c=10.0,
                phase_day=30 if self.latitude >= 0 else 210
            )
            if not self._has_soil_temp_data:
                print(f"Using default Kusuda ground temperature (mean={default_mean:.1f}°C)")

    def generate_weather_data(self, time_hours):
        """
        Generates the interpolated time-series list of weather data.

        Now provides comprehensive weather data matching EPWWeatherLoader output,
        including solar position, sky temperature, and ground temperature.

        Args:
            time_hours (np.array): A numpy array of the simulation time in hours.

        Returns:
            list[dict]: A list of weather data dictionaries with all fields needed
                       for full simulation including opaque solar absorption.
        """
        if self.df_hourly is None:
            raise RuntimeError("Weather data has not been fetched.")

        # 1. Create target DatetimeIndex from time_hours
        start_date_str = self.config['simulation_settings']['start_date']

        # FIX: Localize to UTC.
        # The weather API returns UTC (timezone-aware) timestamps.
        # We must make our simulation start time UTC as well to allow merging/union.
        start_dt = pd.to_datetime(start_date_str).tz_localize('UTC')

        # Convert simulation hours to actual timestamps
        target_index = [start_dt + timedelta(hours=float(t)) for t in time_hours]
        target_dt_index = pd.DatetimeIndex(target_index)

        # 2. Interpolate
        # To interpolate correctly, we combine the hourly index with our target minute-level index,
        # interpolate, and then select only our target points.

        # Union of indices
        combined_index = self.df_hourly.index.union(target_dt_index).sort_values()

        # Reindex to combined, then interpolate time-wise
        # Limit direction='both' handles slight edge cases at start/end
        df_interp = self.df_hourly.reindex(combined_index).interpolate(method='time').ffill().bfill()

        # Select only the target simulation steps
        df_final = df_interp.loc[target_dt_index]

        # 3. Convert to list of dicts matching simulation requirements
        weather_data = []

        # Pre-fetch column series to avoid repeated lookups in loop
        temps = df_final.get('temperature_2m', pd.Series(np.zeros(len(df_final))))
        dew_points = df_final.get('dew_point_2m', pd.Series(np.full(len(df_final), 10.0)))  # Default 10°C
        wind_speeds = df_final.get('wind_speed_10m', pd.Series(np.zeros(len(df_final))))
        wind_dirs = df_final.get('wind_direction_10m', pd.Series(np.zeros(len(df_final))))

        # Solar radiation components
        ghi_values = df_final.get('shortwave_radiation_instant', pd.Series(np.zeros(len(df_final))))
        dni_values = df_final.get('direct_normal_irradiance_instant', pd.Series(np.zeros(len(df_final))))
        dhi_values = df_final.get('diffuse_radiation_instant', pd.Series(np.zeros(len(df_final))))

        # Cloud cover for sky temperature (0-100 scale from API)
        cloud_covers = df_final.get('cloud_cover', pd.Series(np.zeros(len(df_final))))

        # Soil temperature data (if available from Open Meteo)
        # Use surface layer (0-7cm) for ground contact temperature
        soil_temp_surface = df_final.get('soil_temperature_0_to_7cm', None)
        soil_temp_shallow = df_final.get('soil_temperature_7_to_28cm', None)
        soil_temp_medium = df_final.get('soil_temperature_28_to_100cm', None)
        soil_temp_deep = df_final.get('soil_temperature_100_to_255cm', None)

        for i in range(len(df_final)):
            # Get interpolated values
            temp = float(temps.iloc[i])
            dew_point = float(dew_points.iloc[i])
            ghi = max(0.0, float(ghi_values.iloc[i]))
            dni = max(0.0, float(dni_values.iloc[i]))
            dhi = max(0.0, float(dhi_values.iloc[i]))
            cloud_cover = float(cloud_covers.iloc[i])

            # Calculate solar position for this timestep
            # Convert to naive datetime for solar calculator (uses local time logic)
            timestamp = target_dt_index[i].to_pydatetime().replace(tzinfo=None)
            zenith_deg, azimuth_deg, extraterrestrial, is_sunlit = \
                self.solar_calc.calculate_sun_position(timestamp)

            # Calculate sky temperature using Berdahl-Martin model with dew point + cloud cover
            air_temp_k = temp + 273.15
            dew_point_k = dew_point + 273.15
            cloud_fraction = cloud_cover / 100.0  # Convert 0-100 to 0-1
            sky_temp_k = self.sky_temp_calc.calculate_sky_temperature(
                air_temp_k=air_temp_k,
                dew_point_temp_k=dew_point_k,
                cloud_cover_fraction=cloud_fraction
            )

            # Get ground temperature - prefer measured soil temp, fall back to Kusuda model
            if self._has_soil_temp_data and soil_temp_surface is not None:
                # Use measured surface soil temperature (0-7cm depth)
                # This is most relevant for slab-on-grade floor heat transfer
                measured_soil_temp = float(soil_temp_surface.iloc[i])
                if not np.isnan(measured_soil_temp):
                    ground_temp_c = measured_soil_temp
                else:
                    # NaN value - fall back to Kusuda
                    day_of_year = timestamp.timetuple().tm_yday
                    ground_temp_c = self.ground_temp_calc.get_temperature(day_of_year)
            else:
                # No soil temp data - use Kusuda-Achenbach model
                day_of_year = timestamp.timetuple().tm_yday
                ground_temp_c = self.ground_temp_calc.get_temperature(day_of_year)

            weather_data.append({
                'air_temp_c': temp,
                'dew_point_c': dew_point,
                'wind_speed_local_ms': float(wind_speeds.iloc[i]),
                'wind_speed_10m_ms': float(wind_speeds.iloc[i]),
                'wind_direction_deg': float(wind_dirs.iloc[i]),
                'solar_irradiance_w_m2': ghi,  # Legacy GHI field
                'solar_ghi_w_m2': ghi,  # Global horizontal irradiance
                'solar_dni_w_m2': dni,  # Direct normal irradiance
                'solar_dhi_w_m2': dhi,  # Diffuse horizontal irradiance
                'sun_zenith_deg': zenith_deg,
                'sun_azimuth_deg': azimuth_deg,
                'extraterrestrial_w_m2': extraterrestrial,
                'is_sunlit': is_sunlit,
                'sky_temp_k': sky_temp_k,
                'ground_temp_c': ground_temp_c
            })

        return weather_data

# --- Weather Generator Factory ---

WEATHER_GENERATOR_MAP = {
    "simple_sinusoidal": SimpleSinusoidal,
    "file": WeatherFromFile,
    "epw": EPWWeatherLoader
}

def get_weather_generator(config):
    """
    Factory function to get the appropriate weather generator instance.
    
    Args:
        config (dict): The FULL simulation configuration dictionary.
                       (Changed from just 'weather' section to support accessing 
                       sibling keys like 'location' and 'simulation_settings')

    Returns:
        An instance of a weather generator class.
    """
    # Extract weather specific config
    weather_config = config.get("weather", {})
    
    # Default to simple_sinusoidal if 'type' key is missing
    weather_type = weather_config.get("type", "simple_sinusoidal") 
    
    GeneratorClass = WEATHER_GENERATOR_MAP.get(weather_type)
    
    if not GeneratorClass:
        raise ValueError(f"Unknown weather generator type: '{weather_type}'. "
                         f"Available types are: {list(WEATHER_GENERATOR_MAP.keys())}")
    
    # Instantiate
    if weather_type == "simple_sinusoidal":
        # Simple model only needs the weather section
        return GeneratorClass(weather_config)
    elif weather_type == "file":
        # File model needs the full config to access Location and Settings
        return GeneratorClass(config)
    elif weather_type == "epw":
        # EPW model needs the EPW file path from weather config
        epw_path = weather_config.get('epw_path')
        if not epw_path:
            raise ValueError("EPW weather type requires 'epw_path' in weather config")
        return GeneratorClass(epw_path)
    else:
        return GeneratorClass(config)