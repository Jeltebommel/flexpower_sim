# merge_sources.py

import pandas as pd
import os
from pathlib import Path

# Define data directory
DATA_DIR = Path("../data")
LOAD_DIR = Path("../total_load_data")

# Load electricity prices
def load_prices():
    df = pd.read_csv(DATA_DIR / "electricity_prices_2015_2025.csv")
    df["datetime"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.rename(columns={"Price (EUR/MWhe)": "price_eur_mwh"})
    df = df[["datetime", "price_eur_mwh"]]
    return df.set_index("datetime").sort_index()

# Load and merge all load files with dynamic column detection
def load_total_load():
    files = sorted([LOAD_DIR / f for f in os.listdir(LOAD_DIR) if f.lower().endswith(".csv")])
    print("Found load files:", files)
    if not files:
        raise FileNotFoundError("No load files found in total_load_data directory.")
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        # detect datetime column (contains 'time')
        datetime_col = next((c for c in df.columns if 'Time' in c or 'time' in c), None)
        if datetime_col is None:
            raise KeyError(f"No datetime column found in {file}")
        # parse start of interval
        df['datetime'] = pd.to_datetime(df[datetime_col].astype(str).str.split(' - ').str[0], dayfirst=True, errors='coerce')
        # detect forecast and actual columns
        forecast_col = next((c for c in df.columns if 'Forecast' in c or 'forecast' in c), None)
        actual_col = next((c for c in df.columns if 'Actual Total Load' in c or 'actual' in c), None)
        if not forecast_col or not actual_col:
            raise KeyError(f"Forecast or Actual columns not found in {file}")
        df = df.rename(columns={forecast_col: 'forecast_load_mw', actual_col: 'actual_load_mw'})
        df = df[['datetime', 'forecast_load_mw', 'actual_load_mw']]
        df = df.set_index('datetime')
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    return combined

# Load weather data
def load_weather():
    df = pd.read_csv(DATA_DIR / "weather_history_2015_2025.csv")
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "temperature_2m (°C)": "temperature_c",
        "relative_humidity_2m (%)": "humidity_percent",
        "rain (mm)": "rain_mm",
        "snowfall (cm)": "snowfall_cm",
        "wind_speed_10m (km/h)": "wind_speed_10m_kmh",
        "wind_speed_100m (km/h)": "wind_speed_100m_kmh",
        "wind_direction_10m (°)": "wind_dir_10m_deg",
        "wind_direction_100m (°)": "wind_dir_100m_deg",
        "terrestrial_radiation (W/m²)": "terrestrial_radiation",
        "global_tilted_irradiance (W/m²)": "global_tilted_irradiance",
        "direct_normal_irradiance (W/m²)": "direct_normal_irradiance",
        "diffuse_radiation (W/m²)": "diffuse_radiation",
        "direct_radiation (W/m²)": "direct_radiation",
        "shortwave_radiation (W/m²)": "shortwave_radiation",
        "shortwave_radiation_instant (W/m²)": "shortwave_radiation_inst",
        "direct_radiation_instant (W/m²)": "direct_radiation_inst",
        "diffuse_radiation_instant (W/m²)": "diffuse_radiation_inst",
        "direct_normal_irradiance_instant (W/m²)": "dni_inst",
        "global_tilted_irradiance_instant (W/m²)": "gti_inst",
        "terrestrial_radiation_instant (W/m²)": "terrestrial_radiation_inst",
        "cloud_cover (%)": "cloud_cover",
        "cloud_cover_low (%)": "cloud_cover_low",
        "cloud_cover_mid (%)": "cloud_cover_mid",
        "cloud_cover_high (%)": "cloud_cover_high"
    })
    return df.set_index("datetime").sort_index()

# Load TTF gas prices (daily data)
def load_ttf():
    df = pd.read_csv(DATA_DIR / "TTF_price_netherlands.csv")
    # Detect date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        raise KeyError("No date column found in TTF data")
    df["datetime"] = pd.to_datetime(df[date_col])
    # Detect price column (exclude date)
    price_col = next((c for c in df.columns if 'price' in c.lower() and c != date_col.lower()), None)
    if price_col is None:
        raise KeyError("No price column found in TTF data")
    df = df.rename(columns={price_col: "ttf_price"})
    df = df[["datetime", "ttf_price"]].set_index("datetime").sort_index()
    # Resample to hourly by forward-filling daily values
    df = df.resample("H").ffill()
    return df

# Merge all sources together
def merge_all():
    price = load_prices()
    load_df = load_total_load()
    weather = load_weather()
    ttf = load_ttf()

    merged = price.join(load_df, how="left")
    merged = merged.join(weather, how="left")
    merged = merged.join(ttf, how="left")

    merged = merged.sort_index().interpolate().dropna()
    return merged

if __name__ == "__main__":
    df = merge_all()
    print("Merged dataset shape:", df.shape)
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "electricity_training_data.csv")
    print("Saved merged dataset to data/processed/electricity_training_data.csv")
