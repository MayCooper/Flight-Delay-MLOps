# ------------------------------------------------------------------------------
# Author: May Cooper
# Project: Flight Delay Prediction API
#
# Description:
# This FastAPI application provides real-time predictions for average flight
# departure delays. It uses a trained Ridge regression model to estimate delays
# based on the destination airport, departure time, and arrival time.
#
# Key Features:
# - Loads a serialized machine learning model and airport encoding data
# - Validates time input using strict 'HH:MM' formatting
# - Converts input into the same feature structure used during training
# - Exposes RESTful endpoints for prediction and service status
#
# Designed for easy integration with scheduling systems or external tools,
# this API is container-ready and supports scalable deployment.
# ------------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
import json
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Initialize the FastAPI application instance
app = FastAPI()

# Attempt to load airport label encodings used during model training
# If the file doesn't exist, fall back to an empty dictionary and issue a warning
try:
    with open("airport_encodings.json", "r") as f:
        airports = json.load(f)
except FileNotFoundError:
    airports = {}
    print("Warning: airport_encodings.json not found. Defaulting to an empty dictionary.")

# Attempt to load the pre-trained Ridge regression model from disk
# If the file is missing, set the model to None and warn the user
try:
    with open("finalized_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: finalized_model.pkl not found. Predictions will not work.")

# Convert an airport code into a one-hot encoded vector
# The vector has all zeros except for a single 1 at the index associated with the airport
def create_airport_encoding(airport: str, airports: dict) -> np.array:
    vec = np.zeros(len(airports), dtype=float)
    if airport in airports:
        index = airports[airport]
        vec[index] = 1.0
        return vec
    else:
        return None

import re
from fastapi import HTTPException

# Convert a time string in 'HH:MM' format to the number of seconds since midnight
# Includes validation for format correctness and valid hour/minute ranges
def time_to_seconds(time_str: str) -> int:
    pattern = re.compile(r'^\d{2}:\d{2}$')  # Require two-digit hour and minute format

    # Raise an error if the format does not match expected pattern
    if not pattern.match(time_str):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid time format: {time_str}; expected HH:MM with leading zero."
        )

    try:
        hh, mm = map(int, time_str.split(":"))

        # Validate that hour is between 0–23 and minute is between 0–59
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time range: {time_str}. Hours must be 0-23 and minutes 0-59."
            )

        return hh * 3600 + mm * 60
    except ValueError:
        # Raise error if conversion from string to integers fails
        raise HTTPException(
            status_code=400,
            detail=f"Invalid time format: {time_str}; expected HH:MM."
        )

@app.get("/")
def root():
    """
    Root endpoint to confirm that the API is running.
    """
    return {"message": "API is functional"}

# B2
@app.get("/predict/delays")
def predict_average_delay(arrival_airport: str, departure_time: str, arrival_time: str):
    """
    Endpoint to provide average departure delay predictions based on:
      - arrival_airport (str)
      - departure_time (str, format 'HH:MM')
      - arrival_time (str, format 'HH:MM')

    Returns a JSON response indicating the predicted average delay in minutes.
    """
    # Checking if model was loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Cannot predict.")

    # One-hot encode the arrival airport
    airport_vector = create_airport_encoding(arrival_airport, airports)
    if airport_vector is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or unknown arrival airport: {arrival_airport}"
        )

    # Convert departure and arrival times to seconds from midnight
    dep_seconds = time_to_seconds(departure_time)
    arr_seconds = time_to_seconds(arrival_time)

    # Build raw feature vector in the same order as the training from Task 2:
    #    [ one-hot-destination-airports..., hour_depart, hour_arrive ]
    X_raw = np.hstack([
        airport_vector,
        [dep_seconds, arr_seconds]
    ])

    # Replicate the Task 2 polynomial transformation (degree=1)
    poly = PolynomialFeatures(degree=1, include_bias=True)
    X_poly = poly.fit_transform(X_raw.reshape(1, -1))

    # Predict with the loaded Ridge model
    prediction = model.predict(X_poly)
    avg_delay = float(prediction[0])  # single value

    # Return JSON response
    return {
        "arrival_airport": arrival_airport,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "average_delay": round(avg_delay, 2)
    }
