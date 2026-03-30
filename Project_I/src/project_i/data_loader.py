import re
import pandas as pd
import numpy as np


COLUMN_RENAME = {
    "Timestamp": "timestamp",
    'ENERGO ENERGO FVEelektroměr D1_366-A1,A2,A3 - Celkový výkon Celkový výkon kW': "fve_d1_power_kw",
    "ENERGO ENERGO FVE elektroměr Hala_VVN - Celkový výkon Celkový výkon kW": "fve_vvn_power_kw",
    "ENERGO ENERGO Hlavní elektroměr - Okamžitý výkon Okamžitý výkon kW": "main_meter_power_kw",
    "ENERGO ENERGO Meteostanice Slunce střecha  B2 JIH": "solar_irradiance_wm2",
    "ENERGO ENERGO Vývod 19 Monoblok motory - Celkový činný výkon Celkový činný výkon kW": "motors_power_kw",
    "ENERGO ENERGO Vývod 41 Monoblok světla - Celkový činný výkon Celkový činný výkon kW": "lights_power_kw",
    "ENERGO Virtual Sensors Solar Irradiance": "virtual_solar_irradiance",
    "Prague, Hlavni mesto Praha, Czech Republic Daytime": "daytime",
    "Prague, Hlavni mesto Praha, Czech Republic Temp": "temp_c",
}

# Patterns to strip from cell values: unit suffix -> regex
UNIT_PATTERNS = re.compile(r"(kW|_W/mTwo|°C)$")


def _clean_numeric(value):
    """Strip unit suffixes and convert to float. Empty strings become NaN."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value == "" or value == '""':
        return np.nan
    cleaned = UNIT_PATTERNS.sub("", value)
    try:
        return float(cleaned)
    except ValueError:
        return value


def _parse_timestamp(value):
    """Remove ' Prague' suffix and parse ISO 8601 timestamp."""
    if not isinstance(value, str):
        return value
    cleaned = value.replace(" Prague", "")
    return pd.to_datetime(cleaned)


def load_energy_data(filepath):
    """Load the raw energy CSV and return a clean DataFrame.

    - Strips unit suffixes (kW, _W/mTwo, °C) from numeric values
    - Parses timestamps with timezone
    - Renames columns to short English names
    - Sets timestamp as index
    """
    df = pd.read_csv(filepath, dtype=str)

    # Rename columns
    df = df.rename(columns=COLUMN_RENAME)

    # Parse timestamp
    df["timestamp"] = df["timestamp"].apply(_parse_timestamp)
    df = df.set_index("timestamp")

    # Identify numeric columns (everything except daytime)
    numeric_cols = [c for c in df.columns if c != "daytime"]

    for col in numeric_cols:
        df[col] = df[col].apply(_clean_numeric)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean daytime column: empty strings to NaN
    if "daytime" in df.columns:
        df["daytime"] = df["daytime"].replace({"": np.nan, '""': np.nan})

    return df
