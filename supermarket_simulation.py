""" Supermarket simulation
This script reads the data processed by supermarket_analysis_processing.py
at /out to simulate customer behaviour.
"""
import pandas as pd
import numpy as np
import datetime
import cv2
import pickle
import my_constants

from matplotlib import pyplot as plt
from utils import save_obj, load_obj, clear, to_dummy_date, add_minute, state_index_to_vec, state_name_to_index
from customer import Customer, customer_history_processing
from visualize_simulation import Visualize_Simulation
# %% Constants

STATES = my_constants.STATES
TIME_INDEX = my_constants.TIME_INDEX
OUTPUT = True
PLOT = False
VISUALIZE = True

# %% Functions

def generate_first_location_index():
    """
    Generates the first location index using the first_location_prob probabilities 
    derived from real customer data.
    """
    first_location = np.random.choice(
        list(first_location_prob.keys()), p=list(first_location_prob.values())
    )
    return state_name_to_index(first_location)


# %%
if __name__ == "__main__":
    # reads the data processed by supermarket_analysis_processing.py
    transition_matrix = pd.read_csv("out/transition_matrix.csv", index_col=0)
    time_spent_prob = pd.read_csv("out/time_spent_prob.csv", index_col=0)
    df_entertime_freq = pd.read_csv("out/df_entertime_freq.csv", index_col=0)
    people_entering_per_min_mean = df_entertime_freq.mean(axis=1)
    people_entering_per_min_std = df_entertime_freq.std(axis=1)
    first_location_prob = load_obj("first_location_prob")

    # %%
    # Generate the no of customer entering the customer per minute with normal
    # distribution using the mean and std derived from the raw data
    no_of_customer_per_min_list = []
    for i in range(len(people_entering_per_min_mean)):
        no_of_customer_per_min_list.append(
            max(
                round(
                    np.random.normal(
                        people_entering_per_min_mean[i], people_entering_per_min_std[i]
                    )
                ),
                0,
            )
        )

    # Generates a list of customers
    customer_list = []
    for time_index, no_of_customer_per_min in enumerate(no_of_customer_per_min_list):
        for no_of_customer in range(no_of_customer_per_min):
            customer_list.append(
                Customer(
                    generate_first_location_index(),
                    transition_matrix,
                    time_spent_prob,
                    time_index,
                )
            )

    for i, customer in enumerate(customer_list):
        customer.shop()

    df_customer_history, df_customer_history_filled = customer_history_processing(
        customer_list
    )
    # %%
    # --------------------------------------------------------------------------------
    # Plotting simulated customer behaviour
    # --------------------------------------------------------------------------------
    if PLOT:
        \%% Plot the no. of customer at each section
        plt.figure(figsize=(15, 10))
        plt.plot(
            df_customer_history_filled["time"]
            .value_counts()
            .sort_index()
            .rolling(15)
            .mean(),
            label="total",
        )
        plot_df = (
            df_customer_history_filled.groupby(["time"])["location"]
            .value_counts()
            .unstack()
            .fillna(0)
            .rolling(15)
            .mean()
        )
        for i in range(plot_df.shape[1]):
            plt.plot(plot_df.index, plot_df.iloc[:, i], label=plot_df.columns[i])
        plt.legend(loc="upper left")
        title = (
            f"Simulated no. of customer at each section (rolling mean window = 15mins)"
        )
        plt.title(title)
        plt.savefig(f"figure/{title}.png")
        plt.close()

        # %% Plot the amount of time spent at the supermarket
        (
            df_customer_history.groupby(["customer_ID"])["duration"].sum() - 1
        ).value_counts().sort_index().plot.bar(figsize=(15, 10))
        plt.xlabel("Time Spent")
        plt.ylabel("count")
        title = "Simulated count of the amount of time spent at the supermarket"
        plt.title(title)
        plt.savefig(f"figure/{title}.png")
        plt.close()

        # %% Plot the count of the next location given the previous location
        df_customer_history["customer_ID_diff"] = df_customer_history[
            "customer_ID"
        ].diff()
        df_customer_history["previous_location"] = df_customer_history[
            "location"
        ].shift(1)
        locations_to_ignore = [
            "checkout0",
            "checkout1",
            "checkout2",
            "checkout3",
            "checkout4",
            "exit",
        ]
        df_customer_history[
            np.logical_and(
                df_customer_history["customer_ID_diff"] == 0,
                ~df_customer_history["location"].isin(locations_to_ignore),
            )
        ].groupby(["previous_location"])["location"].value_counts().unstack().plot.bar(
            figsize=(15, 10)
        )
        plt.ylabel("count")
        plt.xlabel("Previous location")
        title = "Simulated count of the next location given the previous location"
        plt.title(title)
        plt.savefig(f"figure/{title}.png")
        plt.close()

        # %% Calculate the transition matrix
        sim_transition_matrix = (
            df_customer_history[df_customer_history["customer_ID_diff"] == 0]
            .groupby(["previous_location"])["location"]
            .value_counts()
            .unstack()
        )
        sim_transition_matrix.fillna(0, inplace=True)

        for i in range(sim_transition_matrix.shape[0]):
            sim_transition_matrix.iloc[i] = (
                sim_transition_matrix.iloc[i] / sim_transition_matrix.iloc[i, :].sum()
            )

        temp = pd.DataFrame(
            data={"checkout": 0, "dairy": 0, "drinks": 0, "fruit": 0, "spices": 0},
            index=["checkout"],
        )
        sim_transition_matrix = pd.concat([temp, sim_transition_matrix])

        # %% Plot the revenue
        df_customer_history_filled["revenue"] = df_customer_history_filled[
            "location"
        ].map({"fruit": 4, "spices": 3, "dairy": 5, "drinks": 6})
        df_customer_history_filled[
            ~df_customer_history_filled["location"].isin(locations_to_ignore)
        ].groupby(["location"])["revenue"].sum().plot.bar(figsize=(15, 10))
        plt.ylabel("Revenue")
        plt.xlabel("Section")
        title = f"Simulated Revenue"
        plt.title(title)
        plt.savefig(f"figure/{title}.png")
        plt.close()

        df_customer_history.sort_values(by=["time"])

    # %% Simulation visualization
    if VISUALIZE:
        starting_time = datetime.datetime(2000, 1, 1, 7, 0, 0).time()
        background = cv2.imread("market.png")
        # masks for path finding. mask_exit is to prevent the pathfinding algorithm from going
        # to the cashier in the opposite direction
        mask = cv2.imread("market_maskv2.png")
        mask_exit = cv2.imread("market_mask_exit.png")
        Sim = Visualize_Simulation(
            customer_list, starting_time, background, mask, mask_exit, scale=0.2
        )

        Sim.visualize()

    # %%
    if OUTPUT:
        df_customer_history.drop(
            ["customer_ID_diff", "location_previous"], axis=1, inplace=True
        )
        df_customer_history.reset_index(inplace=True, drop=True)
        df_customer_history.to_csv("simulation_out/df_customer_history.csv")

        df_customer_history_filled.drop(
            ["customer_ID_diff", "location_previous"], axis=1, inplace=True
        )
        df_customer_history_filled.reset_index(inplace=True, drop=True)
        df_customer_history_filled.to_csv(
            "simulation_out/df_customer_history_filled.csv"
        )
