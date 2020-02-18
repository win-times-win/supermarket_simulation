import numpy as np
import pandas as pd
import cv2
import my_constants
import datetime 

from findpath import FindPath
from utils import save_obj, load_obj, clear, to_dummy_date, add_minute, state_index_to_vec, state_name_to_index

STATES = my_constants.STATES
TIME_INDEX = my_constants.TIME_INDEX
location_coordinate = pd.read_csv("out/location_coordinate.csv")

class Customer:
    """
    Class for a single customer that can generate shopping pattern given the initial location,
    speed, the time the customer spends at each section
    """

    def __init__(
        self,
        initial_location,
        transition_matrix,
        time_spent_prob,
        starting_time_index,
        states=STATES,
        time_index=TIME_INDEX,
    ):
        """    
        Parameters
        ----------
        initial_location : integer
            Index of the first location
        transition_matrix : pd.Dataframe
            Transition matrix from out/transition_matrix.csv
        time_spent_prob: pd.DataFrame
            Probability matrix of time spent at each location from out/time_spent_prob.csv
        starting_time_index: integer
            Index of the starting time
        """
        self.current_state_vec = state_index_to_vec(initial_location)
        self.transition_matrix_np = np.asarray(transition_matrix)
        self.time_spent_prob = time_spent_prob
        self.starting_time_index = starting_time_index

        self.states = states
        self.time_index = time_index

        self.history = []
        self.moving_i = 0
        self.shoppingnow = False
        self.moving = False

    def shop(self):
        """Generates a shopping history without queue information for the customer """
        # if the customer is already at the checkout
        if self.current_state_vec[0] == 1:
            return "Customer already shopped"
        duration = 0
        event_no = 0
        current_time_index = self.starting_time_index
        while True:
            location = self.states[self.current_state_vec.index(1)]
            if self.current_state_vec[0] != 1:
                duration_prob = np.array(self.time_spent_prob[location])
                duration = np.random.choice(range(1, 31), p=duration_prob)
                if current_time_index + duration > 900:
                    duration = 900 - current_time_index
                self.history.append(
                    [
                        location,
                        duration,
                        current_time_index,
                        self.time_index.iloc[current_time_index, 0],
                        event_no,
                    ]
                )
            else:
                duration_prob = np.array(self.time_spent_prob[location])
                duration = np.random.choice(range(1, 31), p=duration_prob)
                self.history.append(
                    [
                        location,
                        duration,
                        current_time_index,
                        self.time_index.iloc[current_time_index, 0],
                        event_no,
                    ]
                )
                return "shopped"
            current_time_index += duration
            event_no += 1
            # if the shop is still open
            if current_time_index >= 900:
                current_time_index = 900
                # head to checkout
                self.current_state_vec = state_index_to_vec(0)
            else:
                # calculate the next state probability
                self.current_state_vec = np.matmul(
                    self.current_state_vec, self.transition_matrix_np
                )
                # randomly choose the next state given the probability
                self.current_state_vec = state_index_to_vec(
                    np.random.choice(range(len(self.states)), p=self.current_state_vec)
                )

    def history_with_queue_no(self, customer_ID, df_customer_history):
        """Updates customer history with queue no."""
        i = 0
        self.history = []
        for row_no, row in df_customer_history[
            df_customer_history["customer_ID"] == customer_ID
        ].iterrows():
            self.history.append([])
            self.history[i].append(row["location"])
            self.history[i].append(row["duration"])
            self.history[i].append(
                TIME_INDEX[TIME_INDEX["hour"] == row["time"]].index[0]
            )
            self.history[i].append(row["time"])
            self.history[i].append(row["event_no"])
            self.history[i].append(row["queue_no"])
            i += 1

    def move(self, simulation_time, mask, mask_exit, scale):
        """Checks if there is a new event for the customer, updates the path if there is a new event"""
        time_enter = to_dummy_date(np.array(self.history)[:, 3][0])
        time_exit = to_dummy_date(np.array(self.history)[:, 3][-1])
        if (to_dummy_date(simulation_time) >= time_enter) and (
            to_dummy_date(simulation_time) <= time_exit
        ):  # len(np.array(self.history)[:, 3])-1
            self.shoppingnow = True
            if not self.moving:
                # if there is an event at the current simulation time
                if simulation_time in np.array(self.history)[:, 3]:
                    index = np.where(np.array(self.history)[:, 3] == simulation_time)[
                        0
                    ][0]

                    # coordinate of the door
                    if self.history[index][4] == 0:
                        self.current_coordinate = (770, 700)
                    next_location = np.array(self.history)[index][0]
                    queue_no = np.array(self.history)[index][5]
                    x = location_coordinate[
                        np.logical_and(
                            location_coordinate["location"] == next_location,
                            location_coordinate["queue_no"] == queue_no,
                        )
                    ]["x"].values[0]
                    y = location_coordinate[
                        np.logical_and(
                            location_coordinate["location"] == next_location,
                            location_coordinate["queue_no"] == queue_no,
                        )
                    ]["y"].values[0]
                    next_coordinate = (int(x), int(y))
                    if next_location.find("checkout") != -1:
                        mask = mask_exit
                    findpath = FindPath(mask, scale)
                    self.path_coord = findpath.find(
                        self.current_coordinate, next_coordinate
                    )
                    self.current_coordinate = next_coordinate
                    # initiate moving
                    self.moving = True

                    return (self.shoppingnow, self.moving)
                else:
                    return (self.shoppingnow, self.moving)
            else:
                return (self.shoppingnow, self.moving)
        else:
            self.shoppingnow = False
            return (self.shoppingnow, self.moving)

    def draw(self, simulation_time, layer):
        """Draw the next instance of customer, returns the modified layer."""
        if self.shoppingnow:
            if self.moving:
                if self.moving_i < len(self.path_coord):
                    cv2.circle(
                        layer,
                        (
                            self.path_coord[self.moving_i][0],
                            self.path_coord[self.moving_i][1],
                        ),
                        12,
                        (1, 200, 100),
                        -1,
                    )
                    self.moving_i += 1
                    return layer
                else:
                    self.moving_i = 0
                    self.moving = False
            if not self.moving:
                cv2.circle(
                    layer,
                    (self.current_coordinate[0], self.current_coordinate[1]),
                    12,
                    (1, 200, 100),
                    -1,
                )
                return layer
        else:
            return layer


def find_or_remove_occupancy(occupancy, customer_ID, find=False, remove=False):
    """
    Remove=True: Remove the occupancy of a customer in the occupancy dictionary,
    returns the occupancy dictionary if the customer is not occupying anything
    Find=True: find the index of a customer in the occupancy dictionary,
    returns the occupancy dictionary if the customer is not occupying anything
    """
    out = occupancy
    for key, item in occupancy.items():
        index = np.where(item == customer_ID)[0]
        if len(index) != 0:
            if remove == True:
                out[key][index] = -1
                return out
            if find == True:
                return index
    return out


def customer_history_processing(customer_list):
    """
    Take the customer list, modifies customer queue and returns 2 dataframes:
    1. Customer history without fill
    2. Customer history with fill
    """
    row_list_customer_history = []
    row_list_customer_history_filled = []
    for customer_ID, customer in enumerate(customer_list):
        for history in customer.history:
            row = {
                "customer_ID": customer_ID,
                "event_no": history[4],
                "time": TIME_INDEX.iloc[history[2], 0],
                "duration": history[1],
                "location": history[0],
            }
            row_list_customer_history.append(row)

            for elapsed in range(history[1]):
                # creates a dummy_date for adding minutes
                dummy_date = to_dummy_date(TIME_INDEX.iloc[history[2], 0])
                dummy_date = dummy_date + datetime.timedelta(minutes=elapsed)
                row = {
                    "customer_ID": customer_ID,
                    "event_no": history[4],
                    "time": dummy_date.time(),
                    "location": history[0],
                }
                row_list_customer_history_filled.append(row)

    df_customer_history = pd.DataFrame(row_list_customer_history)
    df_customer_history_filled = pd.DataFrame(row_list_customer_history_filled)

    df_customer_history_filled["customer_ID_diff"] = df_customer_history_filled[
        "customer_ID"
    ].diff()
    df_customer_history_filled["customer_ID_diff"].fillna(1, inplace=True)
    df_customer_history_filled["location_lastmin"] = df_customer_history_filled[
        "location"
    ].shift(1)
    df_customer_history_filled.loc[
        df_customer_history_filled["customer_ID_diff"] == 1, ["location_lastmin"]
    ] = "entrance"
    df_customer_history_filled["queue_no"] = None

    df_customer_history["customer_ID_diff"] = df_customer_history["customer_ID"].diff()
    df_customer_history["customer_ID_diff"].fillna(1, inplace=True)
    df_customer_history["location_previous"] = df_customer_history["location"].shift(1)
    df_customer_history.loc[
        df_customer_history_filled["customer_ID_diff"] == 1, ["location_previous"]
    ] = "entrance"
    df_customer_history["queue_no"] = None

    # Occupancy of each location. -1 if unoccupied, customer_ID if occupied
    # checkout{i}_t is the time counter for the checkout, when value is 0 the item is removed
    # deactivate a checkout by changing -1 to a large number e.g. 5000
    occupancy = {
        "fruit": np.full([20,], -1, dtype=int),
        "drinks": np.full([20,], -1, dtype=int),
        "dairy": np.full([20,], -1, dtype=int),
        "spices": np.full([20,], -1, dtype=int),
        "checkout0": np.full([25,], -1, dtype=int),
        "checkout0_t": np.full([25,], -1, dtype=int),
        "checkout1": np.full([25,], -1, dtype=int),
        "checkout1_t": np.full([25,], -1, dtype=int),
        "checkout2": np.full([25,], -1, dtype=int),
        "checkout2_t": np.full([25,], -1, dtype=int),
        "checkout3": np.full([25,], -1, dtype=int),
        "checkout3_t": np.full([25,], -1, dtype=int),
        "checkout4": np.full([25,], -1, dtype=int),
        "checkout4_t": np.full([25,], -1, dtype=int),
    }

    # updates the df_customer_history_filled to include the queue-no at each location
    for row_no, row in (df_customer_history_filled.sort_values(by=["time"])).iterrows():
        # if there is a change in location
        if row["location_lastmin"] != row["location"]:
            if row["location"] != "checkout":
                rand_no = np.random.choice(
                    np.where(occupancy[row["location"]] == -1)[0]
                )
                occupancy = find_or_remove_occupancy(
                    occupancy, row["customer_ID"], remove=True
                )
                df_customer_history_filled.loc[row_no, "queue_no"] = rand_no
                occupancy[row["location"]][rand_no] = row["customer_ID"]
            else:
                # remove occupancy if customer isat checkout
                occupancy = find_or_remove_occupancy(
                    occupancy, row["customer_ID"], remove=True
                )
        elif row["location"] != "checkout":
            # update the dataframe with the queue_no if the location is same as the one last minute
            df_customer_history_filled.loc[
                row_no, "queue_no"
            ] = find_or_remove_occupancy(occupancy, row["customer_ID"], find=True)

    dict_customer_checkout = {
        "customer_ID": [],
        "event_no": [],
        "time": [],
        "duration": [],
        "location": [],
        "customer_ID_diff": [],
        "location_previous": [],
        "queue_no": [],
    }
    occupancy_t = []
    # loop over every minute to determine the queue at the checkout
    for k, time in enumerate(TIME_INDEX["hour"]):
        occupancy_t.append(occupancy)
        # -1 minute for all checkouts and move the line if the time reaches 0
        for i in range(5):
            occupancy[f"checkout{i}_t"] = occupancy[f"checkout{i}_t"] - 1
            occupancy[f"checkout{i}_t"][occupancy[f"checkout{i}_t"] < 0] = -1
            if occupancy[f"checkout{i}_t"][0] == 0:
                occupancy[f"checkout{i}_t"][:-1] = occupancy[f"checkout{i}_t"][1:]
                occupancy[f"checkout{i}"][:-1] = occupancy[f"checkout{i}"][1:]
            occupancy[f"checkout{i}_t"][occupancy[f"checkout{i}_t"] < 0] = -1

        for row_no, row in df_customer_history[
            np.logical_and(
                df_customer_history["location"] == "checkout",
                df_customer_history["time"] == time,
            )
        ].iterrows():

            checkout_len = np.array(
                [len(np.where(occupancy[f"checkout{i}"] != -1)[0]) for i in range(5)]
            )
            checkout_minlen = np.min(checkout_len)
            # randomly choose amongst the queues with the shortest length
            rand_no = np.random.choice(np.where(checkout_len == checkout_minlen)[0])
            index = np.where(occupancy[f"checkout{rand_no}"] == -1)[0][0]

            occupancy[f"checkout{rand_no}"][index] = row["customer_ID"]
            # if the customer doesn't have to queue then no need to sum up queuing time
            if index == 0:
                occupancy[f"checkout{rand_no}_t"][index] = row["duration"]
            else:
                occupancy[f"checkout{rand_no}_t"][index] = (
                    row["duration"] + occupancy[f"checkout{rand_no}_t"][index - 1]
                )

            # update df_customer_history with queuing info
            previous_duration = 0
            queue_time = row["time"]
            row["location"] = f"checkout{rand_no}"

            for m, n in enumerate(list(range(index, -1, -1))):
                row["queue_no"] = n
                row["event_no"] += m
                row["time"] = add_minute(queue_time, previous_duration)
                row["duration"] = occupancy[f"checkout{rand_no}_t"][m]
                queue_time = add_minute(queue_time, previous_duration)
                previous_duration = occupancy[f"checkout{rand_no}_t"][m].item()
                for key in row.index:
                    dict_customer_checkout[key].append(row[key])

            row["queue_no"] = 0
            row["event_no"] += 1
            row["time"] = add_minute(queue_time, previous_duration)
            row["location"] = "exit"
            row["duration"] = 1
            for key in row.index:
                dict_customer_checkout[key].append(row[key])

    df_customer_checkout = pd.DataFrame(dict_customer_checkout)
    df_customer_history = df_customer_history[
        df_customer_history["location"] != "checkout"
    ]
    df_customer_history = df_customer_history.append(
        df_customer_checkout, ignore_index=True
    )
    df_customer_history["timestamp"] = df_customer_history["time"].apply(to_dummy_date)
    df_customer_history["timestamp"] = pd.to_datetime(df_customer_history["timestamp"])

    # remake a forward fill dataframe
    # iterate through each group, where each group is groupd by date, and customer_no
    temp = []
    df_customer_history.set_index("timestamp", inplace=True)
    for name, group in df_customer_history.sort_values(
        by=["customer_ID", "time"]
    ).groupby(["customer_ID"]):
        # forward resample to each min.
        temp.append(group.resample("min").ffill())
    df_temp = pd.concat(temp)
    df_temp["timestamp"] = df_temp.index
    df_temp["time"] = df_temp["timestamp"].dt.time

    df_customer_history_filled = pd.merge(
        df_temp,
        df_customer_history_filled[["customer_ID", "time", "queue_no"]],
        how="left",
        on=["customer_ID", "time"],
    )
    df_customer_history_filled["queue_no"] = df_customer_history_filled[
        "queue_no_x"
    ].fillna(0) + df_customer_history_filled["queue_no_y"].fillna(0)
    df_customer_history_filled.sort_values(by=["customer_ID", "time"])
    df_customer_history_filled.drop(
        ["timestamp", "queue_no_y", "queue_no_x", "duration"], axis=1, inplace=True
    )

    df_customer_history.drop(["queue_no"], axis=1, inplace=True)
    df_customer_history = pd.merge(
        df_customer_history,
        df_customer_history_filled[["customer_ID", "time", "queue_no"]],
        how="left",
        on=["customer_ID", "time"],
    )
    df_customer_history.sort_values(by=["customer_ID", "time"], inplace=True)

    # update customer history with queue info
    for customer_ID, customer in enumerate(customer_list):
        customer.history_with_queue_no(customer_ID, df_customer_history)

    return df_customer_history, df_customer_history_filled