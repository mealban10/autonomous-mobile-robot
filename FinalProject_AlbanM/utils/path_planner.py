from pathlib import Path

import networkx as nx


class AStarPlanner:

    #---------------------------------------------------------------------------------------------------
    # __init__(...)    Initializes the planner by loading the occupancy grid and building the navigation graph.
    #---------------------------------------------------------------------------------------------------
    def __init__(self, map_file):
        self.map_file = Path(map_file)
        self.grid = self._load_map()
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.graph = self._build_graph()

    #---------------------------------------------------------------------------------------------------
    # _load_map(...)    Loads and validates the occupancy grid from a text file.
    #---------------------------------------------------------------------------------------------------
    def _load_map(self):
        with self.map_file.open("r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            raise ValueError("The map is empty.")

        width = len(lines[0])

        if any(len(line) != width for line in lines):
            raise ValueError("The map rows have different lengths.")

        if any(char not in "012" for line in lines for char in line):
            raise ValueError("The map contains unsupported characters.")

        return [
            [int(char) for char in line]
            for line in lines
        ]

    #---------------------------------------------------------------------------------------------------
    # is_free(...)    Checks whether a grid cell is within the map boundaries and is traversable.
    #---------------------------------------------------------------------------------------------------
    def is_free(self, cell):
        col, row = cell

        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return False

        return self.grid[row][col] == 0

    #---------------------------------------------------------------------------------------------------
    # heuristic(...)    Estimates the distance between two cells using Manhattan distance.
    #---------------------------------------------------------------------------------------------------
    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    #---------------------------------------------------------------------------------------------------
    # _build_graph(...)    Creates a grid graph and removes occupied or blocked cells.
    #---------------------------------------------------------------------------------------------------
    def _build_graph(self):
        graph = nx.grid_2d_graph(self.cols, self.rows)

        blocked = [
            (col, row)
            for row in range(self.rows)
            for col in range(self.cols)
            if not self.is_free((col, row))
        ]

        graph.remove_nodes_from(blocked)
        return graph

    #---------------------------------------------------------------------------------------------------
    # plan(...)    Calculates the shortest valid path between a start cell and a goal using A*.
    #---------------------------------------------------------------------------------------------------
    def plan(self, start, goal):
        if not self.is_free(start):
            raise ValueError(f"Start cell {start} is not free.")

        if not self.is_free(goal):
            raise ValueError(f"Goal cell {goal} is not free.")

        try:
            return nx.astar_path(
                self.graph,
                source=start,
                target=goal,
                heuristic=self.heuristic,
            )
        except nx.NetworkXNoPath:
            return None

    #---------------------------------------------------------------------------------------------------
    # cell_to_world(...)    Converts grid cell coordinates into world coordinates at the cell center.
    #---------------------------------------------------------------------------------------------------
    @staticmethod
    def cell_to_world(cell):
        col, row = cell
        return col + 0.5, row + 0.5