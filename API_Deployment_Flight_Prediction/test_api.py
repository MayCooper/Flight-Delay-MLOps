# ------------------------------------------------------------------------------
# API Test Script: test_predict_delays.py
# Author: May Cooper
# Description:
# This script uses pytest and FastAPI's TestClient to validate the behavior of
# the /predict/delays endpoint. It checks whether the API returns the correct
# HTTP status codes and expected keys in response to various valid and invalid inputs.
#
# Test Cases:
# - A valid request with all correct parameters and formats
# - A malformed time input (e.g., "25:99") to trigger a 400 error
# - A missing required parameter to trigger a 422 validation error
#
# ------------------------------------------------------------------------------

import pytest
from fastapi.testclient import TestClient
from api_python_1_0_0 import app

client = TestClient(app)

@pytest.mark.parametrize(
    "description,params,expected_status",
    [
        (
            "Valid request #1 (standard format)",
            {
                "arrival_airport": "LAX",
                "departure_time": "09:30",
                "arrival_time": "12:00",
            },
            200,
        ),
        (
            "Invalid time format (departure_time='25:99')",
            {
                "arrival_airport": "LAX",
                "departure_time": "25:99",  # Impossible hour & minute
                "arrival_time": "12:00",
            },
            400,
        ),
        (
            "Missing required parameter (arrival_airport)",
            {
                "departure_time": "09:30",
                "arrival_time": "10:15",
            },
            422,
        ),
    ],
)
def test_predict_delays(description, params, expected_status):
    """
    Tests the /predict/delays endpoint with various valid and invalid requests.
    """
    response = client.get("/predict/delays", params=params)
    assert response.status_code == expected_status, f"{description} - {response.text}"

    if expected_status == 200:
        data = response.json()
        assert "average_delay" in data, f"{description} - Missing 'average_delay' key"