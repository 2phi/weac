"""
This module defines the API for the WEAC simulation.

We utilize the FastAPI library to define the API. The FastAPI endpoints will be used for two things:
1. Researchers to send Snowpilot/Snowpack data and run the WEAC simulation.
2. Snow-sport enthusiasts to run the WEAC simulation from the GUI. (In the future included in the WhiteRisk app)

FastAPI syntax is for a route:
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
"""

import fastapi
import logging

logger = logging.getLogger(__name__)

app = fastapi.FastAPI(title="WEAC API", description="API for the WEAC simulation")

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/run_from_file")
def run_from_file():
    logger.info("Running WEAC simulation from file")
    return {"message": "Hello, World!"}

@app.get("/run_from_json_schema")
def run_from_json_schema():
    return {"message": "Hello, World!"}
