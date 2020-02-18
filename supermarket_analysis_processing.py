"""Supermarket data analysis and processing for simulation

This script analyses and processes the customer data from the supermarket.
It outputs plots to /figure if the constant PLOT is set to True.
Outputs processed data to /out for simulation. 
"""
import pandas as pd
import numpy as np
import networkx as nx
import datetime
import pickle
import my_constants
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
from utils import save_obj, load_obj

# %%
# PLOT = True if user wants to output and save the plots
PLOT = True
DAY_DICT = my_constants.DAY_DICT
STATES = my_constants.STATES
TIME_INDEX = my_constants.TIME_INDEX
sns.set()

def _get_markov_edges(Q):
    """gets markov edge for plotting networkx markov chain"""
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges


def markov_plot(transition_matrix):
    """ Creating Markov plot dot file"""
    edges_wts = _get_markov_edges(transition_matrix)

    G = nx.MultiDiGraph()

    STATES = transition_matrix.columns
    G.add_nodes_from(STATES)

    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label="{0:.3f}".format(v))

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw_networkx(G, pos)

    edge_labels = {(n1, n2): d["label"] for n1, n2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.drawing.nx_pydot.write_dot(G, "input.dot")
    #input for bash to convert to png: dot -Tpng input.dot > output.png


# %% Reads raw data
df_1 = pd.read_csv("data/monday.csv", sep=";")
df_2 = pd.read_csv("data/tuesday.csv", sep=";")
df_3 = pd.read_csv("data/wednesday.csv", sep=";")
df_4 = pd.read_csv("data/thursday.csv", sep=";")
df_5 = pd.read_csv("data/friday.csv", sep=";")
df = pd.concat([df_1, df_2, df_3, df_4, df_5])

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.weekday
df["hour"] = df["timestamp"].dt.hour
df["date"] = df["timestamp"].dt.date
df["time"] = df["timestamp"].dt.time

# %% Forward filling every minute with the customer's location
df_ff = df.iloc[0:0].set_index("timestamp")
no_checkout = 0
time_spent = {"fruit": [], "spices": [], "dairy": [], "drinks": []}
first_location = []
temp = []
# iterate through each group, where each group is groupd by date, and customer_no
for name, group in df.sort_values(by=["date", "customer_no", "timestamp"]).groupby(
    ["date", "customer_no"]
):
    first_location.append(group.iloc[0]["location"])
    # get the time spent at each section
    for i in range(group.shape[0] - 1):
        time_spent_at_section = (
            group.iloc[i + 1]["timestamp"] - group.iloc[i]["timestamp"]
        )
        time_spent[group.iloc[i]["location"]].append(time_spent_at_section)
    # forward resample to each min.
    temp.append(group.set_index("timestamp").resample("min").ffill())

df_ff = pd.concat(temp)

# %% Probability of the choice of the first location
first_location_prob = dict(
    zip(
        Counter(first_location).keys(),
        np.array(list(Counter(first_location).values()))
        / sum(Counter(first_location).values()),
    )
)

if PLOT:
    plt.figure(figsize=(15, 10))
    plt.bar(list(first_location_prob.keys()), list(first_location_prob.values()))
    plt.xlabel(f"Section")
    plt.ylabel(f"Probability")
    title = "Probability of the choice of first location "
    plt.title(title)
    plt.savefig(f"figure/{title}.png")
    plt.close()

# %% Count of time spent at each section
time_spent_min = {}
for key, value in time_spent.items():
    to_min = [i.seconds / 60 for i in value]
    time_spent_min[key] = to_min

if PLOT:
    for key in time_spent.keys():
        plt.figure(figsize=(15, 10))
        plt.hist(time_spent_min[key])
        plt.xlabel(f"Time")
        plt.ylabel(f"Count")
        title = f"Count of time spent at {key} section"
        plt.title(title)
        plt.savefig(f"figure/{title}.png")
        plt.close()

# %% Converting the time spent at each section to probabilities
# Create an empty dataframe for probabilities
time_spent_prob = pd.DataFrame(
    columns=["fruit", "spices", "dairy", "drinks"], index=list(range(1, 31))
)
time_spent_prob[:] = 0

# input the counts into the dataframe
for key, value in time_spent_min.items():
    new_column = pd.Series(
        list(Counter(value).values()), name=key, index=list(Counter(value).keys())
    )
    time_spent_prob.update(new_column)

time_spent_prob.replace(to_replace=0, method="ffill", inplace=True)

for column in time_spent_prob.columns:
    time_spent_prob[column] = time_spent_prob[column] / time_spent_prob[column].sum()

# %% Plot the total number of customer at each section
if PLOT:
    df["location"].value_counts().sort_index().plot.bar(figsize=(15, 10))
    plt.xlabel("Location")
    plt.ylabel("Count")
    title = "Total no. of customer at each section from Monday to Friday"
    plt.title(title)
    plt.savefig(f"figure/{title}.png")
    plt.close()

# %% Calculate the no. of customer at each section per minute per day
df_section_customer_no = (
    df_ff.groupby(["timestamp"])["location"].value_counts().unstack().fillna(0)
)
df_section_customer_no["sum"] = df_section_customer_no.sum(axis=1)

if PLOT:
    time_range_to_plot = [
        ["Monday", "2019-09-02 07:00:00", "2019-09-02 22:00:00"],
        ["Tuesday", "2019-09-03 07:00:00", "2019-09-03 22:00:00"],
        ["Wednesday", "2019-09-04 07:00:00", "2019-09-04 " "22:00:00"],
        ["Thursday", "2019-09-05 07:00:00", "2019-09-05 22:00:00"],
        ["Friday", "2019-09-06 07:00:00", "2019-09-06 22:00:00"],
    ]
    # plot No. of customer at each section with a rolling mean
    for time_range in time_range_to_plot:
        df_section_customer_no.rolling(15).mean().loc[
            time_range[1] : time_range[2], :
        ].plot(figsize=(15, 10))
        plt.ylabel("Count")
        plt.xlabel("Time")
        title = f"No. of customer at each section on {time_range[0]} (rolling mean window = 15mins)"
        plt.title(title)
        plt.savefig(f"figure/{title}.png")
        plt.close()

# %% Calculate the amount of time spent by each customer by getting its last timestamp minus its first timestamp
df_timespend = df.groupby(["date", "customer_no"])["timestamp"].max().unstack().fillna(
    0
) - df.groupby(["date", "customer_no"])["timestamp"].min().unstack().fillna(0)
df_timespend = pd.concat(
    [df_timespend.T.iloc[:, i] for i in range(df_timespend.T.shape[1])]
)
df_timespend = pd.DataFrame(df_timespend)
df_timespend.reset_index(drop=True, inplace=True)
df_timespend = df_timespend[df_timespend[0] != 0]

if PLOT:
    for i in range(5):
        (
            df[df["timestamp"].dt.weekday == i]
            .groupby(["customer_no"])["timestamp"]
            .max()
            - df[df["timestamp"].dt.weekday == i]
            .groupby(["customer_no"])["timestamp"]
            .min()
        ).value_counts().sort_index().plot.bar(figsize=(15, 10))
        plt.xlabel("Time Spent")
        plt.ylabel("count")
        title = f"Count of the amount of time spent at the supermarket on {DAY_DICT[i]}"
        plt.title(title)
        plt.savefig(f"figure/{title}.png")
        plt.close()

# %% Histogram of time spent at the supermarket per day
if PLOT:
    (df_timespend[0].value_counts().sort_index()).plot.bar(figsize=(15, 10))
    plt.xlabel("Time Spent")
    plt.ylabel("count")
    title = "Count of the amount of time spent at the supermarket from Monday to Friday"
    plt.title(title)
    plt.savefig(f"figure/{title}.png")
    plt.close()

df_entertime = df.groupby(["date", "customer_no"])["timestamp"].min().unstack().T
df_entertime.head(3)

# %% Calculate the mean and std of number of people entering per minute for the simulation assuming normal distribution
df_entertime_freq = pd.DataFrame(
    columns=["hour"],
    index=pd.date_range("2019-09-02 07:00:00", "2019-09-02 22:00:00", freq="1min"),
)
df_entertime_freq["hour"] = df_entertime_freq.index.time
df_entertime_freq.reset_index(drop=True, inplace=True)

for i in range(df_entertime.shape[1]):
    df_temp = pd.DataFrame(df_entertime.iloc[:, i].value_counts().sort_index())
    df_temp["hour"] = df_temp.index.time
    df_entertime_freq = pd.merge(
        df_entertime_freq, df_temp, how="left", left_on="hour", right_on="hour"
    )

df_entertime_freq.fillna(0, inplace=True)
people_entering_per_min_mean = df_entertime_freq.mean(axis=1)
people_entering_per_min_std = df_entertime_freq.std(axis=1)

# %% Plot the count of the next location given the previous location
df.sort_values(by=["date", "customer_no", "timestamp"], inplace=True)
# customer_no changed if customer_no_diff != 0, same customer if customer_no_diff = 0
df["customer_no_diff"] = df["customer_no"].diff()
df["previous_location"] = df["location"].shift(1)

if PLOT:
    df[df["customer_no_diff"] == 0].groupby(["previous_location"])[
        "location"
    ].value_counts().unstack().plot.bar()
    plt.ylabel("count")
    plt.xlabel("Previous location")
    title = "Count of the next location given the previous location"
    plt.title(title)
    plt.savefig(f"figure/{title}.png")
    plt.close()

# %% Plot revenue
df_ff["revenue"] = df_ff["location"].map(
    {"fruit": 4, "spices": 3, "dairy": 5, "drinks": 6}
)
df_ff.reset_index(inplace=True)
df_ff["weekday"] = df_ff["timestamp"].dt.weekday

if PLOT:
    for i in range(5):
        df_ff[
            np.logical_and(df_ff["location"] != "checkout", df_ff["weekday"] == i)
        ].groupby(["location"])["revenue"].sum().plot.bar()
        plt.ylabel("Revenue")
        plt.xlabel("Section")
        title = f"Revenue on {DAY_DICT[i]}"
        plt.title(title)
        plt.savefig(f"figure/{title}.png")
        plt.close()

    df_ff[df_ff["location"] != "checkout"].groupby(["location"])[
        "revenue"
    ].sum().plot.bar()
    plt.ylabel("Revenue")
    plt.xlabel("Section")
    title = f"Revenue from Monday to Friday"
    plt.title(title)
    plt.savefig(f"figure/{title}.png")
    plt.close()

# %% Calculate transition matrix
transition_matrix = (
    df[df["customer_no_diff"] == 0]
    .groupby(["previous_location"])["location"]
    .value_counts()
    .unstack()
)
transition_matrix.fillna(0, inplace=True)

for i in range(transition_matrix.shape[0]):
    transition_matrix.iloc[i] = (
        transition_matrix.iloc[i] / transition_matrix.iloc[i, :].sum()
    )

temp = pd.DataFrame(
    data={"checkout": 0, "dairy": 0, "drinks": 0, "fruit": 0, "spices": 0},
    index=["checkout"],
)
transition_matrix = pd.concat([temp, transition_matrix])

# %% <save as csv or pickle for simulation>
df.to_csv("out/df.csv")
df_ff.to_csv("out/df_ff.csv")
df_entertime_freq.to_csv("out/df_entertime_freq.csv")
transition_matrix.to_csv("out/transition_matrix.csv")
time_spent_prob.to_csv("out/time_spent_prob.csv")
save_obj(first_location_prob, "first_location_prob")
