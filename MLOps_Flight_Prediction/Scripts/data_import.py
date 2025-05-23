# ------------------------------------------------------------------------------
# Author: May Cooper
# Script: data_import.py
#
# Imports a CSV file, renames key columns to match the ML pipeline format,
# validates required fields, and saves the formatted output.
# ------------------------------------------------------------------------------

import pandas as pd
import os
import sys

def import_and_format_data(file_path):
    """
    Import data from a CSV file, format it for processing, and validate required columns.
    """
    required_columns = {
        'YEAR': 'YEAR',
        'MONTH': 'MONTH',
        'DAY_OF_MONTH': 'DAY',
        'DAY_OF_WEEK': 'DAY_OF_WEEK',
        'ORIGIN': 'ORG_AIRPORT',
        'DEST': 'DEST_AIRPORT',
        'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE',
        'DEP_TIME': 'DEPARTURE_TIME',
        'DEP_DELAY': 'DEPARTURE_DELAY',
        'CRS_ARR_TIME': 'SCHEDULED_ARRIVAL',
        'ARR_TIME': 'ARRIVAL_TIME',
        'ARR_DELAY': 'ARRIVAL_DELAY'
    }

    try:
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Rename columns to fit the model's expected format
        data.rename(columns=required_columns, inplace=True)

        # Check if all required columns by the model are present
        missing_columns = [col for col in required_columns.values() if col not in data.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None

        print(f"Data imported and formatted successfully with {len(data)} rows and {len(data.columns)} columns.")

        # Construct the output file path
        output_dir = os.path.dirname(file_path)  # Directory of the input file
        output_filename = "formatted_" + os.path.basename(file_path)  # Add "formatted_" to the input filename
        output_file = os.path.join(output_dir, output_filename)  # Combine directory and filename

        # Save formatted data
        data.to_csv(output_file, index=False)
        print(f"Formatted data saved to {output_file}")

        return data
    except Exception as e:
        print(f"Error importing data: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_import.py <file_path>")
    else:
        file_path = sys.argv[1]
        import_and_format_data(file_path)
