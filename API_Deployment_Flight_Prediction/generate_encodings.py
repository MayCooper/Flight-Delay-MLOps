import pandas as pd
import json

# Load the CSV file
csv_file = "T_ONTIME_REPORTING.csv"
df = pd.read_csv(csv_file)

# Extract unique airport codes from ORIGIN and DEST columns
unique_airports = pd.concat([df["ORIGIN"], df["DEST"]]).unique()

# Create a dictionary mapping each airport code to a unique index
airport_encodings = {airport: index for index, airport in enumerate(unique_airports)}

# Save the dictionary as a JSON file
output_file = "airport_encodings.json"
with open(output_file, "w") as f:
    json.dump(airport_encodings, f, indent=4)

print(f"Airport encodings saved to {output_file}")
