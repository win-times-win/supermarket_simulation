""" my_constants
This script contains the constants for the simulations.
"""
import pandas as pd

STATES = ["checkout", "dairy", "drinks", "fruit", "spices"]

TIME_INDEX = pd.DataFrame(
    columns=["hour"],
    index=pd.date_range("2019-09-02 07:00:00", "2019-09-02 23:59:00", freq="1min"),
)
TIME_INDEX["hour"] = TIME_INDEX.index.time
TIME_INDEX.reset_index(drop=True, inplace=True)

DAY_DICT = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}