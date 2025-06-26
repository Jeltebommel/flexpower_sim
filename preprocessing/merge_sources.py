
import pandas as pd
import glob
import os
from pathlib import Path

# Define data directory
DATA_DIR = Path("data")

# Load electricity prices
def load_prices():
    df = pd.read_csv(DATA_DIR / "electricity_prices_2015_2025.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()

# Load and merge all load files
def load_total_load():
    files = sorted(glob.glob(str(DATA_DIR / "Total Load - Day Ahead _Actual_*.csv")))
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        dfs.append(df.set_index("datetime"))
    return pd.concat(dfs).sort_index()

# Load weather data
def load_weather():
    df = pd.read_csv(DATA_DIR / "weather_history_2015_2025.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()

# Load TTF gas prices (daily data)
def load_ttf():
    df = pd.read_csv(DATA_DIR / "TTF_price_netherlands.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    # Resample to hourly by forward-filling daily values
    df = df.resample("H").ffill()
    return df

# Merge all sources together
def merge_all():
    price = load_prices()
    load = load_total_load()
    weather = load_weather()
    ttf = load_ttf()

    merged = price
    merged = merged.join(load, how="left")
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
