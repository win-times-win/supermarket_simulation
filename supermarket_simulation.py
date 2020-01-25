""" Supermarket simulation
This script reads the data processed by supermarket_analysis_processing.py
at /out to simulate customer behaviour.
"""
import pandas as pd
import numpy as np
import datetime
import cv2
import pickle

from matplotlib import pyplot as plt
from os import system, name

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# %% Constants
STATES = ["checkout", "dairy", "drinks", "fruit", "spices"]

TIME_INDEX = pd.DataFrame(
    columns=["hour"],
    index=pd.date_range("2019-09-02 07:00:00", "2019-09-02 23:59:00", freq="1min"),
)
TIME_INDEX["hour"] = TIME_INDEX.index.time
TIME_INDEX.reset_index(drop=True, inplace=True)

OUTPUT = True
PLOT = False
VISUALIZE = True

# %% Functions
def clear():
    """ clears the screen"""
    # for windows
    if name == "nt":
        _ = system("cls")

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")


def save_obj(obj, name):
    """saves object into pickle file """
    with open("out/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """load object from pickle file """
    with open("out/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


def state_index_to_vec(state_index, no_of_states=len(STATES)):
    """ Takes the state_index as input, outputs a vector that is all zero except the state_index which is 1."""
    vec = [0] * no_of_states
    vec[state_index] = 1
    return vec


def state_name_to_index(state_name, states=STATES):
    """Takes the state_name as input, outputs its index on the list 'STATE'."""
    return states.index(state_name)


def generate_first_location_index():
    """
    Generates the first location index using the first_location_prob probabilities 
    derived from real customer data.
    """
    first_location = np.random.choice(
        list(first_location_prob.keys()), p=list(first_location_prob.values())
    )
    return state_name_to_index(first_location)


def to_dummy_date(time_in):
    """Returns a dummy date in datetime.datetime for arithmetic"""
    return datetime.datetime(2000, 1, 1, time_in.hour, time_in.minute, 0)


def add_minute(time_in, min_add):
    """Takes in a datetime.time and add minutes to it, returns a datetime.time"""
    return (to_dummy_date(time_in) + datetime.timedelta(minutes=min_add)).time()


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


# %%
class FindPath:
    """
    A class that uses the pathfinding module Astar algorithm to find the 
    optimal path in a numpy matrix. 0 indicates obstable, 1 indicates walkable.
    """

    def __init__(self, mask, scale):
        """    
        Parameters
        ----------
        mask : numpy.array
            A numpy array of 1 and 0
        scale : float
            The scale of the mask with respect to the original image
        """
        self.mask = np.pad(mask, [(0, 54), (0, 0)], mode="constant", constant_values=3)
        self.grid = Grid(matrix=self.mask)
        self.scale = scale

    def find(self, start_coor, end_coor):
        """Input start and end coordinate, returns the optimal path"""
        start_coor = (
            round(start_coor[0] * self.scale),
            round(start_coor[1] * self.scale),
        )
        end_coor = round(end_coor[0] * self.scale), round(end_coor[1] * self.scale)

        start = self.grid.node(start_coor[0], start_coor[1])
        end = self.grid.node(end_coor[0], end_coor[1])

        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, runs = finder.find_path(start, end, self.grid)

        path = np.array(path)
        path = (np.array(path) / self.scale).astype(int)
        return path


# %%
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


class Visualize_Simulation:
    """
    Class of tools for visualizing the simulation.
    """

    def __init__(
        self, customer_list, starting_time, background, mask, mask_exit, scale=0.2
    ):
        """    
        Parameters
        ----------
        customer_list : list
            List of customer objects
        starting_time : Datetime.time
            Starting time of the simulation
        background: image
            Supermarket background
        mask: image
            Supermarket mask for pathfinding
        mask_exit: image
            Supermarket mask for pathfinding when heading to checkout
        """
        self.customer_list = customer_list
        self.starting_time = starting_time
        self.background = background

        # resize mask for a more efficient Astar pathfinding
        self.mask = cv2.resize(
            mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.threshold(self.mask, 1, 1, cv2.THRESH_BINARY)[1]

        self.mask_exit = cv2.resize(
            mask_exit, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        self.mask_exit = cv2.cvtColor(self.mask_exit, cv2.COLOR_BGR2GRAY)
        self.mask_exit = cv2.threshold(self.mask_exit, 1, 1, cv2.THRESH_BINARY)[1]

        self.scale = scale

    def visualize(self):
        """Start visualization"""
        i = 0
        self.current_time = self.starting_time

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (1, 1, 1)
        lineType = 2

        while True:
            layer = np.zeros((self.background.shape[0], self.background.shape[1], 3))
            frame = self.background.copy()
            if i % 200 == 0:
                self.current_time = add_minute(self.starting_time, i / 200)
                for self.customer_ID, customer in enumerate(self.customer_list):
                    customer.move(
                        self.current_time, self.mask, self.mask_exit, self.scale
                    )
            for self.customer_ID, customer in enumerate(self.customer_list):
                layer = customer.draw(self.current_time, layer)

            cv2.putText(
                layer,
                self.current_time.strftime("%H:%M:%S"),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
            # filter to only display pixels that are not black
            cnd = layer[:] > 0
            frame[cnd] = layer[cnd]

            cv2.imshow("frame", frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


# %%
if __name__ == "__main__":
    # reads the data processed by supermarket_analysis_processing.py
    transition_matrix = pd.read_csv("out/transition_matrix.csv", index_col=0)
    time_spent_prob = pd.read_csv("out/time_spent_prob.csv", index_col=0)
    location_coordinate = pd.read_csv("out/location_coordinate.csv")
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
    # --------------------------------------------------------------------------------
    # Plotting simulated customer behaviour
    # --------------------------------------------------------------------------------
    if PLOT:
        #%% Plot the no. of customer at each section
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
