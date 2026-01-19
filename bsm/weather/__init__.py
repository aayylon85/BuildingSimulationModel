"""
Weather data handling.

This module contains weather data generators and utilities:
- EPWWeatherLoader: Load weather from EPW files
- WeatherFromFile: Load weather from Open-Meteo API or EPW files
- SkyTemperatureCalculator: Sky temperature calculations
- KusudaGroundTemperature: Ground temperature model
"""

from bsm.weather.generators import (
    EPWWeatherLoader,
    WeatherFromFile,
    SimpleSinusoidal,
    get_weather_generator,
)
from bsm.weather.api_client import get_hourly_weather
from bsm.weather.sky_temperature import SkyTemperatureCalculator
from bsm.weather.ground_temperature import KusudaGroundTemperature

__all__ = [
    "EPWWeatherLoader",
    "WeatherFromFile",
    "SimpleSinusoidal",
    "get_weather_generator",
    "get_hourly_weather",
    "SkyTemperatureCalculator",
    "KusudaGroundTemperature",
]
