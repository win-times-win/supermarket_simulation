import numpy as np
import pandas as pd
import cv2
import my_constants

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