# ------------------------------------------------------------------------------
# Author: May Cooper
# Script: filter_clean_mia.py
#
# Filters flight data for MIA (Miami Florida) departures, removes outliers and missing values,
# and saves a cleaned dataset for modeling.
# ------------------------------------------------------------------------------

import pandas as pd
import sys

def filter_and_clean_mia(file_path):
    """
    Filter data for departures from MIA airport and clean it.
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Filter for MIA airport (Miami, Florida)
        data['ORG_AIRPORT'] = data['ORG_AIRPORT'].astype(str).str.strip().str.upper()
        filtered = data[data['ORG_AIRPORT'] == 'MIA']

        # Drop rows with missing delays
        filtered = filtered.dropna(subset=['DEPARTURE_DELAY', 'ARRIVAL_DELAY'])

        # Remove extreme delays
        filtered['DEPARTURE_DELAY'] = filtered['DEPARTURE_DELAY'].astype(float)
        filtered['ARRIVAL_DELAY'] = filtered['ARRIVAL_DELAY'].astype(float)
        filtered = filtered[filtered['DEPARTURE_DELAY'] <= 1000]

        # Remove flight distances that are less than 50 miles
        if 'DISTANCE' in filtered.columns:
            filtered = filtered[filtered['DISTANCE'] >= 50]

        # Add a new Boolean-value column for on-time flights (ON_TIME)
        filtered['ON_TIME'] = (filtered['DEPARTURE_DELAY'] <= 0) & (filtered['ARRIVAL_DELAY'] <= 0)

        # Remove flights with missing flight numbers
        if 'OP_CARRIER_FL_NUM' in filtered.columns:
            filtered = filtered.dropna(subset=['OP_CARRIER_FL_NUM'])

        # Remove flights with unreasonable taxi times above 500 minutes, which are unrealistic
        if 'TAXI_OUT' in filtered.columns:
            filtered = filtered[filtered['TAXI_OUT'] <= 500]
        if 'TAXI_IN' in filtered.columns:
            filtered = filtered[filtered['TAXI_IN'] <= 500]

        # Check flight numbers are properly formatted to integer where applicable
        if 'OP_CARRIER_FL_NUM' in filtered.columns:
            filtered['OP_CARRIER_FL_NUM'] = filtered['OP_CARRIER_FL_NUM'].astype(str).str.strip()

        # Check if filtered data is empty
        if filtered.empty:
            print("No rows found for MIA.")
        else:
            # Save the cleaned data
            output_file = "cleaned_data.csv"  # Updated output filename
            filtered.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}. Rows: {len(filtered)}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_clean_mia.py <file_path>")
    else:
        filter_and_clean_mia(sys.argv[1])