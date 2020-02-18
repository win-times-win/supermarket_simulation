""" utils
This script contains utilities.
"""
import pickle
import datetime
import my_constants
from os import system, name

STATES = my_constants.STATES
TIME_INDEX = my_constants.TIME_INDEX


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


def to_dummy_date(time_in):
    """Returns a dummy date in datetime.datetime for arithmetic"""
    return datetime.datetime(2000, 1, 1, time_in.hour, time_in.minute, 0)


def add_minute(time_in, min_add):
    """Takes in a datetime.time and add minutes to it, returns a datetime.time"""
    return (to_dummy_date(time_in) + datetime.timedelta(minutes=min_add)).time()


def state_index_to_vec(state_index, no_of_states=len(STATES)):
    """ Takes the state_index as input, outputs a vector that is all zero except the state_index which is 1."""
    vec = [0] * no_of_states
    vec[state_index] = 1
    return vec


def state_name_to_index(state_name, states=STATES):
    """Takes the state_name as input, outputs its index on the list 'STATE'."""
    return states.index(state_name)
