import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


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