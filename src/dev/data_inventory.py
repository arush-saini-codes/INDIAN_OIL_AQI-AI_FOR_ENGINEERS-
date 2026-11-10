"""Data Inventory Script for UCS321 EST Project.

Analyzes every data file in data/raw/ and generates comprehensive metadata,
missingness profiles, location coverage, and AQI availability.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)


def analyze_file(filepath):
    rel_path = os.path.relpath(
        filepath, os.path.dirname(os.path.dirname(filepath))
    )
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print("=" * 80)
    print(f"FILE: {rel_path}")
    print(f"Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")

    # Read data
    df = pd.read_csv(filepath)
    n_rows, n_cols = df.shape
    print(f"Dimensions: {n_rows:,} rows x {n_cols} columns")

    # Column names & dtypes
    print("\nColumns & Data Types:")
    dtype_dict = {}
    for col, dt in df.dtypes.items():
        dtype_dict[col] = str(dt)
        print(f"  - {col:<20}: {dt}")

    # Missing value percentages
    print("\nMissing Values per Column:")
    missing_pct = (df.isnull().sum() / n_rows * 100).round(2)
    for col, pct in missing_pct.items():
        print(f"  - {col:<20}: {pct:>6.2f}% ({df[col].isnull().sum():,} nulls)")

    # Date / Time columns
    date_cols = [
        c
        for c in df.columns
        if "date" in c.lower() or "time" in c.lower() or "datetime" in c.lower()
    ]
    print(f"\nDate / Time Columns Detected: {date_cols}")
    granularity = "Unknown"
    date_info = {}
    for dc in date_cols:
        sample_vals = df[dc].dropna().head(3).tolist()
        # Parse to datetime
        try:
            parsed = pd.to_datetime(df[dc].dropna())
            min_date = str(parsed.min())
            max_date = str(parsed.max())
            date_info[dc] = {
                "sample": sample_vals,
                "min": min_date,
                "max": max_date,
            }
            print(f"  Column: {dc}")
            print(f"    Sample values: {sample_vals}")
            print(f"    Date Range   : {min_date} to {max_date}")

            if "hour" in dc.lower() or "time" in dc.lower():
                granularity = "Hourly"
            elif "day" in filepath.lower() or "day" in dc.lower():
                granularity = "Daily"
            elif ":" in str(sample_vals[0]):
                granularity = "Hourly / Sub-daily"
            else:
                granularity = "Daily"
        except Exception as e:
            print(f"    Failed to parse dates in {dc}: {e}")

    if not date_cols:
        granularity = "Metadata / Static (No time series)"

    print(f"Estimated Granularity: {granularity}")

    # Location columns
    loc_cols = [
        c
        for c in df.columns
        if c.lower()
        in ["city", "station", "stationid", "stationname", "state", "status"]
    ]
    print(f"\nLocation Columns Detected: {loc_cols}")
    for lc in loc_cols:
        n_unique = df[lc].nunique()
        print(f"  Location Field '{lc}': {n_unique} unique values")
        top_counts = df[lc].value_counts()
        print(f"    Top locations by count:")
        for loc_name, count in top_counts.head(10).items():
            print(f"      - {loc_name:<25}: {count:,} rows")
        if n_unique <= 30:
            print(f"    All unique values in '{lc}':")
            for loc_name, count in top_counts.items():
                print(f"      - {loc_name:<25}: {count:,} rows")

    # AQI availability
    has_aqi = "AQI" in df.columns
    has_aqi_bucket = "AQI_Bucket" in df.columns
    pollutant_cols = [
        c
        for c in [
            "PM2.5",
            "PM10",
            "NO",
            "NO2",
            "NOx",
            "NH3",
            "CO",
            "SO2",
            "O3",
            "Benzene",
            "Toluene",
            "Xylene",
        ]
        if c in df.columns
    ]

    print("\nAQI & Pollutants Assessment:")
    print(
        f"  Direct AQI Column Present?    : {'YES (column: AQI)' if has_aqi else 'NO'}"
    )
    print(
        f"  AQI_Bucket Column Present?   : {'YES (column: AQI_Bucket)' if has_aqi_bucket else 'NO'}"
    )
    print(
        f"  Pollutant Columns Detected ({len(pollutant_cols)}): {pollutant_cols}"
    )
    if not has_aqi and pollutant_cols:
        print(
            "  NOTE: File contains pollutant features only; AQI would need sub-index computation or mapping."
        )

    print("\n")


def main():
    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
    ]
    files.sort()
    print(f"Total raw data files found: {len(files)}")
    for f in files:
        analyze_file(f)


if __name__ == "__main__":
    main()

