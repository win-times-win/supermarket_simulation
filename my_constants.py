import pandas as pd

STATES = ["checkout", "dairy", "drinks", "fruit", "spices"]

TIME_INDEX = pd.DataFrame(
    columns=["hour"],
    index=pd.date_range("2019-09-02 07:00:00", "2019-09-02 23:59:00", freq="1min"),
)
TIME_INDEX["hour"] = TIME_INDEX.index.time
TIME_INDEX.reset_index(drop=True, inplace=True)