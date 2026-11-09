import csv
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pybullet as p
import networkx as nx

import utils.world as W
import utils.demo_robot_2w as R
import utils.demo_person as P

from utils.odometry import DifferentialDriveOdometry
from utils.controller_integrated import MotionController
from utils.path_planner import AStarPlanner
from utils.task_manager import Job, TaskManager
from utils.fleet_manager import FleetManager, Robot as PlannedRobot

# Localization modules
from utils.particle_filter import ParticleFilter
from utils.ScanableMap import ScanableMap
from utils.dynamic_obstacle_tracker import DynamicObstacleTracker


# --------------------------------------------------------------
# User configuration: select a scenario and edit its settings
# --------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

# Choose which warehouse experiment to run: "scenario_1" or "scenario_2".
SCENARIO_ID = "scenario_1"

# Both scenarios currently use the same map. Each scenario has its own
# robot starting positions, movement speeds, and pickup/delivery jobs.
SCENARIOS = {
    "scenario_1": {
        "name": "Scenario 1",
        "map_file": "maps/demoMap",
        "robot_starts": {
            "Robot 1": (1, 0),
            "Robot 2": (23, 17),
        },
        "drive_speed": 3,
        "turn_speed": 1.5,
        "jobs": [
            {"job_id": "Job A", "pickup": (3, 2),   "delivery": (5, 2)},
            {"job_id": "Job B", "pickup": (20, 15), "delivery": (15, 17)},
            {"job_id": "Job C", "pickup": (11, 14), "delivery": (7, 16)},
            {"job_id": "Job D", "pickup": (7, 9),   "delivery": (9, 9)},
        ],
        # "time" minimizes the longest estimated robot route.
        # "movements" minimizes the total number of grid movements.
        "optimization_mode": "time",
        "people_enabled": False,
        # Per-scenario robot actuator noise (multiplicative wheel-speed noise).
        # Enable/disable dynamic-obstacle tracking and its safety actions.
        "dynamic_obstacles_enabled": False,
    },
    "scenario_2": {
        "name": "Scenario 2",
        "map_file": "maps/demoMap",
        "robot_starts": {
            "Robot 1": (1, 0),
            "Robot 2": (23, 17),
        },
        "drive_speed": 3,
        "turn_speed": 1.5,
        "jobs": [
            {"job_id": "Job A", "pickup": (3, 2), "delivery": (7, 5)},
            {"job_id": "Job B", "pickup": (20, 15), "delivery": (15, 12)},
            {"job_id": "Job C", "pickup": (11, 14), "delivery": (9, 9)},
            {"job_id": "Job D", "pickup": (7, 9), "delivery": (17, 9)},
        ],
        # "time" minimizes the longest estimated robot route.
        # "movements" minimizes the total number of grid movements.
        "optimization_mode": "time",
        "people_enabled": True,
        # Per-scenario robot actuator noise (multiplicative wheel-speed noise).
        # Enable/disable dynamic-obstacle tracking and its safety actions.
        "dynamic_obstacles_enabled": False,
    },
    "scenario_3": {
        "name": "Scenario 3",
        "map_file": "maps/demoMap",
        "robot_starts": {
            "Robot 1": (4, 2),
            "Robot 2": (20, 11),
        },
        "drive_speed": 3,
        "turn_speed": 1.5,
        "jobs": [
            {"job_id": "Job A", "pickup": (10, 10), "delivery": (18, 13)},
            {"job_id": "Job B", "pickup": (18, 13), "delivery": (10, 10)},
        ],
        # "time" minimizes the longest estimated robot route.
        # "movements" minimizes the total number of grid movements.
        "optimization_mode": "time",
        "people_enabled": True,
        # Enable/disable dynamic-obstacle tracking and its safety actions.
        "dynamic_obstacles_enabled": True,
    },
    "scenario_4": {
        "name": "Scenario 4",
        "map_file": "maps/demoMap",
        "robot_starts": {
            "Robot 1": (13, 9),
            "Robot 2": (16, 9),
        },
        "drive_speed": 1.5,
        "turn_speed": 1.5,
        "jobs": [
            {
                "job_id": "Job A",
                "pickup": (13, 9),
                "delivery": (16, 9),
            },
            {
                "job_id": "Job B",
                "pickup": (16, 9),
                "delivery": (13, 9),
            },
        ],
        "optimization_mode": "time",
        "people_enabled": False,
        "dynamic_obstacles_enabled": True,
    },
}

# General simulation timing and navigation accuracy settings.
PHYSICS_DT = 1.0 / 240.0
CONTROL_DT = 1.0 / 60.0
VISUALIZATION_INTERVAL = 48
POSITION_TOLERANCE = 0.05
ANGLE_TOLERANCE_DEG = 2.0
MAX_ATTEMPTS = 5

# General physical wheel-motion noise settings (shared by all scenarios).
WHEEL_NOISE_ENABLED = False
WHEEL_MOTION_NOISE = 0.05

# Odometry sensor settings used in the localization experiment.
ODOMETRY_NOISE_ENABLED = False
ODOMETRY_QUANTIZATION_ENABLED = False
ENCODER_TICKS_PER_REV = 360
ODOMETRY_WHEEL_ANGLE_NOISE_STD = 0.001

# Colors used to identify each robot in route plots and PyBullet.
ROBOT_COLORS = {
    "Robot 1": [0.1, 0.5, 1.0],
    "Robot 2": [1.0, 0.5, 0.0],
}

# Heights of route lines and pickup/delivery markers in PyBullet.
ROUTE_HEIGHT = 0.04
MARKER_HEIGHT = 0.12

# Create the result folders and file names for the selected experiment.
if SCENARIO_ID not in SCENARIOS:
    raise ValueError(
        f"Unknown scenario '{SCENARIO_ID}'. Choose one of: "
        f"{', '.join(SCENARIOS)}."
    )

ACTIVE_SCENARIO = SCENARIOS[SCENARIO_ID]
MAP_FILE = PROJECT_DIR / ACTIVE_SCENARIO["map_file"]
ROBOT_STARTS = ACTIVE_SCENARIO["robot_starts"]
DRIVE_SPEED = ACTIVE_SCENARIO["drive_speed"]
TURN_SPEED = ACTIVE_SCENARIO["turn_speed"]
JOBS = [
    Job(
        item["job_id"],
        pickup=tuple(item["pickup"]),
        delivery=tuple(item["delivery"]),
    )
    for item in ACTIVE_SCENARIO["jobs"]
]
OPTIMIZATION_MODE = ACTIVE_SCENARIO["optimization_mode"]
PEOPLE_ENABLED = ACTIVE_SCENARIO["people_enabled"]
DYNAMIC_OBSTACLES_ENABLED = ACTIVE_SCENARIO["dynamic_obstacles_enabled"]

RESULTS_DIR = PROJECT_DIR / "results"
RESULT_DIRS = {
    "loc": RESULTS_DIR / "loc",
    "plan": RESULTS_DIR / "plan",
    "dyn": RESULTS_DIR / "dyn",
    "integ": RESULTS_DIR / "integ",
}
for result_dir in RESULT_DIRS.values():
    result_dir.mkdir(parents=True, exist_ok=True)

SCENARIO_PREFIX = SCENARIO_ID
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
LOCALIZATION_CSV_FILE = RESULT_DIRS["loc"] / f"{SCENARIO_PREFIX}_{RUN_ID}_localization.csv"
PLAN_CSV_FILE = RESULT_DIRS["plan"] / f"{SCENARIO_PREFIX}_{RUN_ID}_jobs.csv"
ROUTE_IMAGE_FILE = RESULT_DIRS["plan"] / f"{SCENARIO_PREFIX}_{RUN_ID}_routes_{OPTIMIZATION_MODE}.png"
OBSTACLE_CSV_FILE = RESULT_DIRS["dyn"] / f"{SCENARIO_PREFIX}_{RUN_ID}_obstacle_events.csv"
SUMMARY_CSV_FILE = RESULT_DIRS["integ"] / f"{SCENARIO_PREFIX}_{RUN_ID}_summary.csv"

RUN_START_REAL = time.perf_counter()

# --------------------------------------------------------------
# Particle Filter configuration
# --------------------------------------------------------------

NUM_PARTICLES = 80

# Run the measurement update every N control ticks.
# Prediction still runs on every control tick.
PF_UPDATE_EVERY = 8

# Must match the physical LiDAR configuration in DemoRobot2W.
LIDAR_RAYS = 42
LIDAR_FOV = 2.0
LIDAR_MAX_DISTANCE = 10.0

# The occupancy grid resolution is 10 cells per world unit.
SCAN_MAP_SCALE = 10

# Draw the lines of the particle filter
DRAW_PARTICLES = False

# way of localization: "particle_filter" or "odometry"
LOCALIZATION_MODE = "particle_filter"

# --------------------------------------------------------------
# Dynamic obstacle tracking and safety configuration
# --------------------------------------------------------------

DYNAMIC_OBSTACLE_THRESHOLD = 0.30
DYNAMIC_OBSTACLE_MAX_DISTANCE = 3.5

# Wait briefly before replanning around a persistent obstacle.
BLOCKED_REPLAN_DELAY = 1.5
DYNAMIC_CELL_CLEARANCE = 0.75
REPLAN_RETRY_INTERVAL = 1.0

# LiDAR visualization
LIDAR_LINES = True
LIDAR_DISPLAY_DISTANCE = 1.8
LIDAR_SLOW_DISTANCE = 1.8
LIDAR_STOP_DISTANCE = 0.9

# --------------------------------------------------------------
# Coordinate conversion helpers
# --------------------------------------------------------------

def world_to_cell(position):
    """Convert world coordinates to a grid cell."""
    x, y = position
    return int(math.floor(x)), int(math.floor(y))


# --------------------------------------------------------------
# Matplotlib visualization
# --------------------------------------------------------------

def plot_plan(
    planner,
    plans,
    robot_starts,
    output_file,
    display_paths_by_robot=None,
):
    """Display the planned routes and job markers in 2D."""
    grid = np.array(planner.grid)
    occupancy = (grid != 0).astype(int)

    plt.ion()

    fig, ax = plt.subplots(figsize=(13, 9))

    ax.imshow(
        occupancy,
        cmap=ListedColormap(["white", "dimgray"]),
        origin="upper",
        interpolation="nearest",
    )

    route_artists = {}
    job_artists = {}

    for plan in plans:
        robot_id = plan["robot_id"]
        color = ROBOT_COLORS.get(
            robot_id,
            [0.2, 0.8, 0.3],
        )
        # Display the exact same simplified route used by the controller.
        if display_paths_by_robot is not None:
            path = display_paths_by_robot.get(
                plan["robot_id"],
                plan["path"],
            )
        else:
            path = plan["path"]

        path_x = [col for col, row in path]
        path_y = [row for col, row in path]

        route_line, = ax.plot(
            path_x,
            path_y,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=2.5,
            label=f'{robot_id}: {", ".join(plan["jobs"])}',
        )

        route_artists[robot_id] = route_line

        for job_route in plan["job_routes"]:
            job_id = job_route["job_id"]

            pickup = job_route["to_pickup"][-1]
            delivery = job_route["to_delivery"][-1]

            pickup_artist = ax.scatter(
                *pickup,
                marker="s",
                s=110,
                color=color,
                edgecolors="black",
                zorder=4,
            )

            delivery_artist = ax.scatter(
                *delivery,
                marker="*",
                s=190,
                color=color,
                edgecolors="black",
                zorder=4,
            )

            pickup_label = ax.annotate(
                f"{job_id} pickup",
                pickup,
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )

            delivery_label = ax.annotate(
                f"{job_id} delivery",
                delivery,
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )

            job_artists[
                (robot_id, job_id, "pickup")
            ] = (pickup_artist, pickup_label)

            job_artists[
                (robot_id, job_id, "delivery")
            ] = (delivery_artist, delivery_label)

    for robot_id, start in robot_starts.items():
        ax.scatter(
            start[0],
            start[1],
            marker="o",
            s=130,
            color="limegreen",
            edgecolors="black",
            zorder=5,
        )

        ax.annotate(
            robot_id,
            start,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xticks(np.arange(-0.5, planner.cols, 1))
    ax.set_yticks(np.arange(-0.5, planner.rows, 1))

    ax.grid(color="lightgray", linewidth=0.5)

    ax.set_xlim(-0.5, planner.cols - 0.5)
    ax.set_ylim(planner.rows - 0.5, -0.5)
    ax.set_aspect("equal")

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Warehouse: robot routes and pickup/delivery")

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show(block=False)
    plt.pause(0.1)

    return fig, ax, route_artists, job_artists


def remove_matplotlib_job_marker(job_artists, key):
    """Remove a completed pickup or delivery from the 2D plot."""
    artists = job_artists.pop(key, None)

    if artists is not None:
        marker, label = artists
        marker.remove()
        label.remove()

# --------------------------------------------------------------
# PyBullet visualization helpers
# --------------------------------------------------------------

def draw_route(path, color):
    """Draw the planned route as debug lines in PyBullet."""
    line_ids = []

    for start, end in zip(path[:-1], path[1:]):
        line_id = p.addUserDebugLine(
            [start[0], start[1], ROUTE_HEIGHT],
            [end[0], end[1], ROUTE_HEIGHT],
            lineColorRGB=color,
            lineWidth=3,
            lifeTime=0,
        )

        line_ids.append(line_id)

    return line_ids

def compress_grid_path(path):
    """Keep only turns and the final cell of a grid path."""
    if not path:
        return []
    if len(path) <= 2:
        return list(path)

    result = [tuple(path[0])]
    for i in range(1, len(path) - 1):
        prev_cell = path[i - 1]
        cell = path[i]
        next_cell = path[i + 1]
        direction_in = (cell[0] - prev_cell[0], cell[1] - prev_cell[1])
        direction_out = (next_cell[0] - cell[0], next_cell[1] - cell[1])
        if direction_in != direction_out:
            result.append(tuple(cell))
    result.append(tuple(path[-1]))
    return result

def plan_around_tracks(planner, start_cell, goal_cell, tracks):
    """Run A* on a temporary graph with confirmed dynamic tracks blocked."""
    graph = planner.graph.copy()
    blocked = set()

    for track in tracks:
        if track.get("consecutive_hits", 0) < 2:
            continue
        tx, ty = track["x"], track["y"]
        # Block grid-cell centers that fall within the person clearance.
        for row in range(planner.rows):
            for col in range(planner.cols):
                cell = (col, row)
                if planner.is_free(cell):
                    cx, cy = planner.cell_to_world(cell)
                    if math.hypot(cx - tx, cy - ty) <= DYNAMIC_CELL_CLEARANCE:
                        blocked.add(cell)

    # Never remove the robot's current cell or its active goal.
    blocked.discard(tuple(start_cell))
    blocked.discard(tuple(goal_cell))
    graph.remove_nodes_from(blocked)

    if start_cell not in graph or goal_cell not in graph:
        return None
    try:
        return nx.astar_path(
            graph, source=tuple(start_cell), target=tuple(goal_cell),
            heuristic=planner.heuristic,
        )
    except nx.NetworkXNoPath:
        return None

def draw_job_marker(cell, job_id, marker_type, color):
    """Create a visual marker and label for a job waypoint."""
    x = cell[0] + 0.5
    y = cell[1] + 0.5

    if marker_type == "pickup":
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.13, 0.13, 0.025],
            rgbaColor=[*color, 1.0],
        )
        label = f"{job_id} PICKUP"

    else:
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.14,
            rgbaColor=[*color, 1.0],
        )
        label = f"{job_id} DELIVERY"

    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=[x, y, MARKER_HEIGHT],
    )

    text_id = p.addUserDebugText(
        label,
        [x, y, MARKER_HEIGHT + 0.18],
        textColorRGB=color,
        textSize=1.2,
        lifeTime=0,
    )

    return {
        "body_id": body_id,
        "text_id": text_id,
        "cell": cell,
        "type": marker_type,
        "job_id": job_id,
    }


def remove_job_marker(marker):
    """Remove a completed job marker from PyBullet."""
    p.removeBody(marker["body_id"])
    p.removeUserDebugItem(marker["text_id"])


def update_completed_markers(
    robot_id,
    reached_cell,
    job_markers,
    matplotlib_job_artists,
):
    """Remove job markers when their assigned robot reaches them."""
    for key, marker in list(job_markers.items()):
        marker_robot, job_id, marker_type = key

        if marker_robot != robot_id:
            continue

        if marker["cell"] != reached_cell:
            continue

        remove_job_marker(marker)
        del job_markers[key]

        remove_matplotlib_job_marker(
            matplotlib_job_artists,
            key,
        )

        print(
            f"{robot_id}: completed {job_id} "
            f"{marker_type}"
        )

# --------------------------------------------------------------
# Activate LIDAR lines 
# --------------------------------------------------------------

def draw_lidar_rays(
    robot_pose,
    distances,
    angles,
    safety_state="CLEAR",
    z=0.15,
    ray_width=1.5,
    lifetime=0.2,
):
    if not LIDAR_LINES:
        return

    x, y, theta = robot_pose

    # Select color based on the robot safety state
    if safety_state == "STOP":
        color = [1, 0, 0]       # Red
    elif safety_state == "SLOW":
        color = [1, 0.5, 0]     # Orange
    else:
        color = [0, 0.4, 1]     # Blue

    for distance, angle in zip(distances, angles):

        if not np.isfinite(distance) or distance <= 0:
            continue

        display_distance = min(
            distance,
            LIDAR_DISPLAY_DISTANCE
        )

        world_angle = angle

        end_x = x + display_distance * math.cos(world_angle)
        end_y = y + display_distance * math.sin(world_angle)

        p.addUserDebugLine(
            [x, y, z],
            [end_x, end_y, z],
            lineColorRGB=color,
            lineWidth=ray_width,
            lifeTime=lifetime,
        )

# --------------------------------------------------------------
# Plan all jobs before starting the physical simulation
# --------------------------------------------------------------

planner = AStarPlanner(MAP_FILE)


def validate_scenario(planner, robot_starts, jobs, drive_speed, turn_speed):
    """Check scenario settings before starting the simulation."""
    if drive_speed <= 0 or turn_speed <= 0:
        raise ValueError("Drive speed and turn speed must both be greater than zero.")
    if not robot_starts:
        raise ValueError("The selected scenario must define at least one robot.")

    def require_free_cell(cell, description):
        cell = tuple(cell)
        if not planner.is_free(cell):
            raise ValueError(
                f"{description} {cell} is outside the map or is not a free cell. "
                "Check the selected scenario settings."
            )

    for robot_id, start_cell in robot_starts.items():
        require_free_cell(start_cell, f"Starting cell for {robot_id}")

    job_ids = [job.job_id for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Job IDs must be unique within each scenario.")

    for job in jobs:
        require_free_cell(job.pickup, f"Pickup cell for {job.job_id}")
        require_free_cell(job.delivery, f"Delivery cell for {job.job_id}")


validate_scenario(planner, ROBOT_STARTS, JOBS, DRIVE_SPEED, TURN_SPEED)
task_manager = TaskManager(JOBS)

planned_robots = [
    PlannedRobot(robot_id, start_cell)
    for robot_id, start_cell in ROBOT_STARTS.items()
]

fleet_manager = FleetManager(
    planner,
    task_manager,
    planned_robots,
)

plans, best_score = fleet_manager.choose_best_assignment(
    task_manager.pending_jobs(),
    optimization_mode=OPTIMIZATION_MODE,
)

fleet_manager.commit_assignment(plans)

with open(PLAN_CSV_FILE, "w", newline="", encoding="utf-8") as plan_file:
    plan_writer = csv.writer(plan_file)
    plan_writer.writerow([
        "scenario_id", "robot_id", "job_id",
        "pickup_col", "pickup_row", "delivery_col", "delivery_row",
        "planned_route_movements",
    ])
    jobs_by_id = {job.job_id: job for job in JOBS}
    for plan in plans:
        for job_route in plan["job_routes"]:
            job = jobs_by_id[job_route["job_id"]]
            plan_writer.writerow([
                SCENARIO_ID, plan["robot_id"], job.job_id,
                job.pickup[0], job.pickup[1],
                job.delivery[0], job.delivery[1],
                job_route["cost"],
            ])

print(f"Scenario: {ACTIVE_SCENARIO['name']} ({SCENARIO_ID})")
print(f"Map: {MAP_FILE}")
print(f"Map dimensions: {planner.cols} columns x {planner.rows} rows")
print(f"Optimization mode: {OPTIMIZATION_MODE}")
print("\nPlanned assignment:")

for plan in plans:
    print(
        f'  {plan["robot_id"]}: '
        f'{", ".join(plan["jobs"])} '
        f'({plan["cost"]} movements)'
    )

print(f"Estimated completion cost: {best_score} movements")


# Convert each grid route into physical waypoint coordinates.
plans_by_robot = {
    plan["robot_id"]: plan
    for plan in plans
}

waypoints_by_robot = {}
simplified_grid_paths_by_robot = {}
simplified_world_paths_by_robot = {}

for robot_id in ROBOT_STARTS:
    plan = plans_by_robot.get(robot_id)

    if plan is None or not plan["path"]:
        waypoints_by_robot[robot_id] = []
        simplified_grid_paths_by_robot[robot_id] = []
        simplified_world_paths_by_robot[robot_id] = []
        continue

    grid_path = plan["path"]
    world_path = [
        planner.cell_to_world(cell)
        for cell in grid_path
    ]

    # Keep pickup and delivery cells as mandatory stops. Compress only
    # intermediate cells that lie on the same straight grid segment.
    mandatory_cells = set()
    for job_route in plan.get("job_routes", []):
        for key in ("to_pickup", "to_delivery"):
            segment = job_route.get(key, [])
            if segment:
                mandatory_cells.add(tuple(segment[-1]))

    simplified_world_path = []
    if grid_path:
        simplified_world_path.append(world_path[0])

        for index in range(1, len(grid_path) - 1):
            previous_cell = grid_path[index - 1]
            current_cell = grid_path[index]
            next_cell = grid_path[index + 1]

            direction_in = (
                current_cell[0] - previous_cell[0],
                current_cell[1] - previous_cell[1],
            )
            direction_out = (
                next_cell[0] - current_cell[0],
                next_cell[1] - current_cell[1],
            )

            is_corner = direction_in != direction_out
            is_job_stop = tuple(current_cell) in mandatory_cells

            if is_corner or is_job_stop:
                simplified_world_path.append(world_path[index])

        if len(grid_path) > 1:
            simplified_world_path.append(world_path[-1])

    # Store the same simplified route in grid and world coordinates.
    # The first point is the robot's start; the remaining points are waypoints.
    simplified_grid_path = []
    if grid_path:
        simplified_grid_path.append(tuple(grid_path[0]))

        for index in range(1, len(grid_path) - 1):
            previous_cell = grid_path[index - 1]
            current_cell = grid_path[index]
            next_cell = grid_path[index + 1]

            direction_in = (
                current_cell[0] - previous_cell[0],
                current_cell[1] - previous_cell[1],
            )
            direction_out = (
                next_cell[0] - current_cell[0],
                next_cell[1] - current_cell[1],
            )

            if (
                direction_in != direction_out
                or tuple(current_cell) in mandatory_cells
            ):
                simplified_grid_path.append(tuple(current_cell))

        if len(grid_path) > 1:
            simplified_grid_path.append(tuple(grid_path[-1]))

    simplified_grid_paths_by_robot[robot_id] = simplified_grid_path
    simplified_world_paths_by_robot[robot_id] = simplified_world_path
    waypoints_by_robot[robot_id] = simplified_world_path[1:]


# --------------------------------------------------------------
# Create the shared PyBullet world
# --------------------------------------------------------------

world = W.BaseWorld()
world.map_from_file(str(MAP_FILE))
world.texture_walls()

world.physics_dt = PHYSICS_DT
p.setTimeStep(PHYSICS_DT)

# Create moving people only when they are enabled for this scenario.
if PEOPLE_ENABLED:
    person1 = P.DemoPerson()
    person2 = P.DemoPerson()
    person3 = P.DemoPerson()

    person1.set_position(7, 2)
    person2.set_position(4, 8.5)
    person3.set_position(10, 11)

    # Define the movement paths followed by each person.
    g1 = P.ValidPathGraph()
    g1.add_positions([(7, 2.5), (7, 4.5)])
    g1.add_connection((7, 2.5), (7, 4.5))

    g2 = P.ValidPathGraph()
    g2.add_positions([(10.5, 4.5), (12.5, 4.5)])
    g2.add_connection((10.5, 4.5), (12.5, 4.5))

    g3 = P.ValidPathGraph()
    g3.add_positions([(20, 10.5), (20, 12.5)])
    g3.add_connection((20, 10.5), (20, 12.5))

    person1.set_movement_graph(g1)
    person2.set_movement_graph(g2)
    person3.set_movement_graph(g3)

    # Register each person's movement callback with the simulation.
    world.add_step_callback(person1.step_action)
    world.add_step_callback(person2.step_action)
    world.add_step_callback(person3.step_action)

print(f"Moving people enabled: {PEOPLE_ENABLED}")
print(f"Wheel motion noise: enabled={WHEEL_NOISE_ENABLED}, magnitude={WHEEL_MOTION_NOISE:.3f}")
print(f"Dynamic obstacle detection enabled: {DYNAMIC_OBSTACLES_ENABLED}")

# Display the planned routes and markers in Matplotlib.
fig, ax, route_artists, matplotlib_job_artists = plot_plan(
    planner,
    plans,
    ROBOT_STARTS,
    ROUTE_IMAGE_FILE,
    display_paths_by_robot=simplified_grid_paths_by_robot,
)

# Draw the same routes and job markers in PyBullet.
route_lines = {}
job_markers = {}

for plan in plans:
    robot_id = plan["robot_id"]
    color = ROBOT_COLORS.get(
        robot_id,
        [0.2, 0.8, 0.3],
    )

    # Draw exactly one line segment per controller waypoint.
    # This keeps route_lines[i] aligned with waypoints[i].
    world_path = simplified_world_paths_by_robot[robot_id]
    route_lines[robot_id] = draw_route(world_path, color)

    for job_route in plan["job_routes"]:
        job_id = job_route["job_id"]

        pickup_cell = job_route["to_pickup"][-1]
        delivery_cell = job_route["to_delivery"][-1]

        job_markers[
            (robot_id, job_id, "pickup")
        ] = draw_job_marker(
            pickup_cell,
            job_id,
            "pickup",
            color,
        )

        job_markers[
            (robot_id, job_id, "delivery")
        ] = draw_job_marker(
            delivery_cell,
            job_id,
            "delivery",
            color,
        )


# --------------------------------------------------------------
# Create the shared scan map for the Particle Filters
# --------------------------------------------------------------

# Both robots use the same static occupancy map.
# Each robot will have its own independent ParticleFilter.
#
# The map dimensions are given in world units, not grid cells.
# The FOV is converted from radians to degrees because
# ScanableMap expects degrees in its constructor.

scan_map = ScanableMap(
    map_dimensions=(planner.cols, planner.rows),
    fov=np.degrees(LIDAR_FOV),
    max_distance=LIDAR_MAX_DISTANCE,
    num_rays=LIDAR_RAYS,
    scale=SCAN_MAP_SCALE,
)

scan_map.load_map(str(MAP_FILE))


# --------------------------------------------------------------
# Create the physical robots, odometry, PFs and controllers
# --------------------------------------------------------------

robots = {}

for robot_id, start_cell in ROBOT_STARTS.items():

    # Convert the start grid cell into world coordinates.
    start_x, start_y = planner.cell_to_world(start_cell)
    start_theta = 0.0

    # Create the physical robot in PyBullet.
    robot = R.DemoRobot2W(
        x=start_x,
        y=start_y,
        motion_noise=WHEEL_MOTION_NOISE,
        noise_enabled=WHEEL_NOISE_ENABLED,
    )

    # Create wheel-based odometry.
    odometry = DifferentialDriveOdometry(
        wheel_radius=0.12,
        wheel_base=0.56,
        left_wheel_joint=1,
        right_wheel_joint=0,
        noise_enabled=ODOMETRY_NOISE_ENABLED,
        quantization_enabled=ODOMETRY_QUANTIZATION_ENABLED,
        encoder_ticks_per_rev=ENCODER_TICKS_PER_REV,
        wheel_angle_noise_std=ODOMETRY_WHEEL_ANGLE_NOISE_STD,
    )

    odometry.reset(
        start_x,
        start_y,
        start_theta,
    )

    # Establish the initial wheel encoder reference.
    odometry.update(robot.agv_id)

    # Create one independent particle filter per robot.
    particle_filter = ParticleFilter(
        num_particles=NUM_PARTICLES,
        scan_map=scan_map,
    )

    particle_filter.initialize(
        start_x,
        start_y,
        start_theta,
    )

    # The initial PF estimate is the known starting pose.
    pf_pose = (start_x, start_y, start_theta)

    # Create the navigation controller.
    controller = MotionController(
        robot=robot,
        world=world,
        odometry=odometry,
        drive_speed=DRIVE_SPEED,
        turn_speed=TURN_SPEED,
        position_tolerance=POSITION_TOLERANCE,
        angle_tolerance_deg=ANGLE_TOLERANCE_DEG,
        timeout=120.0,

        # Keep odometry active while validating the PF.
        localization_mode=LOCALIZATION_MODE,
    )

    # Give the controller the initial PF estimate.
    # This will be used if LOCALIZATION_MODE is "particle_filter".
    controller.set_particle_pose(*pf_pose)

    # Store all state for this robot together.
    robots[robot_id] = {
        "robot": robot,
        "odometry": odometry,
        "particle_filter": particle_filter,
        "controller": controller,

        "pf_pose": pf_pose,
        "pf_weights": np.ones(NUM_PARTICLES) / NUM_PARTICLES,
        "control_ticks": 0,

        # Each robot keeps independent obstacle tracks and safety state.
        "obstacle_tracker": DynamicObstacleTracker(
            distance_threshold=DYNAMIC_OBSTACLE_THRESHOLD,
            max_detection_distance=DYNAMIC_OBSTACLE_MAX_DISTANCE,
            association_distance=0.8,
            max_track_age=1.0,
            min_group_rays=2,
            front_angle=LIDAR_FOV / 2,
        ),
        "obstacle_stop": False,
        "obstacle_speed_scale": 1.0,
        "last_safety_state": "clear",
        "blocked_since": None,
        "last_replan_attempt": -float("inf"),
        "replan_count": 0,

        "waypoints": waypoints_by_robot[robot_id],
        "waypoint_index": 0,

        "failed": False,
        "finished": False,
    }


# --------------------------------------------------------------
# Shared simulation clock and CSV logging
# --------------------------------------------------------------

clock = {
    "physics_steps": 0,
    "control_accumulator": 0.0,
}

localization_file = None
localization_writer = None
obstacle_file = None
obstacle_writer = None

# --------------------------------------------------------------
# Control callback: odometry + PF + navigation + logging
# --------------------------------------------------------------

def control_callback():
    """Update odometry, particle filters and controllers."""

    # Track simulation time using the physics clock.
    clock["physics_steps"] += 1

    sim_time = clock["physics_steps"] * world.physics_dt

    clock["control_accumulator"] += world.physics_dt

    # Run the robot controllers at CONTROL_DT, not every physics step.
    if clock["control_accumulator"] + 1e-12 < CONTROL_DT:
        return

    clock["control_accumulator"] -= CONTROL_DT

    # Update each robot independently.
    for robot_id, state in robots.items():

        robot = state["robot"]
        odometry = state["odometry"]
        particle_filter = state["particle_filter"]
        controller = state["controller"]

        # ----------------------------------------------------
        # 1. Update odometry and obtain motion increments
        # ----------------------------------------------------

        # delta_s is the estimated forward movement since the
        # previous odometry update.
        #
        # delta_theta is the estimated heading change.
        delta_s, delta_theta = odometry.update(robot.agv_id)

        # ----------------------------------------------------
        # 2. Predict particle poses using odometry
        # ----------------------------------------------------

        # Each particle is propagated using the same measured
        # robot motion, with noise added inside ParticleFilter.
        particle_filter.predict(
            delta_s,
            delta_theta,
        )

        state["control_ticks"] += 1

        # ----------------------------------------------------
        # 3. Correct the PF using the real LiDAR measurements
        # ----------------------------------------------------

        if state["control_ticks"] % PF_UPDATE_EVERY == 0:

            # Obtain LiDAR distances and world-frame ray angles
            # from the physical robot in PyBullet.
            real_distances, real_angles = robot.ray_cast_lidar(
                num_rays=LIDAR_RAYS,
                fov=LIDAR_FOV,
                max_distance=LIDAR_MAX_DISTANCE,
            )

            # Calculate how well each particle's predicted scan
            # matches the real scan.
            weights = particle_filter.calculate_weights(
                real_distances,
                real_angles,
            )

            # Estimate the robot pose from the weighted particles.
            pf_x, pf_y, pf_theta = (
                particle_filter.estimate_pose(weights)
            )

            # Dynamic-obstacle detection is controlled per scenario.
            # PF localization continues to run even when this feature is off.
            if DYNAMIC_OBSTACLES_ENABLED:
                expected_distances, _ = scan_map.scan(
                    pf_x,
                    pf_y,
                    pf_theta,
                )

                tracks, safety = state["obstacle_tracker"].update(
                    real_distances=real_distances,
                    real_angles=real_angles,
                    expected_distances=expected_distances,
                    robot_pose=(pf_x, pf_y, pf_theta),
                    sim_time=sim_time,
                )
                state["obstacle_stop"] = safety["stop"]
                state["obstacle_speed_scale"] = safety["speed_scale"]

                safety_state = (
                    "STOP" if safety["stop"]
                    else "SLOW" if safety["speed_scale"] < 1.0
                    else "CLEAR"
                )
                if safety_state != state["last_safety_state"]:
                    if safety_state != "CLEAR":
                        print(
                            f"{robot_id}: obstacle safety={safety_state}; "
                            f"tracked candidates={len(tracks)}"
                        )
                    if obstacle_writer is not None:
                        obstacle_writer.writerow([
                            SCENARIO_ID,
                            sim_time,
                            time.perf_counter() - RUN_START_REAL,
                            robot_id,
                            safety_state,
                            len(tracks),
                            safety["speed_scale"],
                        ])
                    state["last_safety_state"] = safety_state

                lidar_visualization_enabled= LIDAR_LINES and ACTIVE_SCENARIO["dynamic_obstacles_enabled"]

                # Draw LiDAR rays using the actual tracker decisions.
                if lidar_visualization_enabled:
                    draw_lidar_rays(
                        robot_pose=(pf_x, pf_y, pf_theta),
                        distances=real_distances,
                        angles=real_angles,
                        safety_state=safety["state"],
                    )

            else:
                # Explicitly clear any stale stop/slow command when disabled.
                state["obstacle_stop"] = False
                state["obstacle_speed_scale"] = 1.0
                state["last_safety_state"] = "DISABLED"

            state["pf_pose"] = (pf_x, pf_y, pf_theta)

            state["pf_weights"] = weights

            # Keep the controller's PF pose up to date.
            controller.set_particle_pose(
                pf_x,
                pf_y,
                pf_theta,
            )

            # Draw particles for debugging.
            # This is optional because drawing many debug lines
            # repeatedly can slow down the simulation.
            if DRAW_PARTICLES:
                particle_filter.draw_particles(weights)

            # Resample after estimating the pose.
            # The next prediction starts from the resampled set.
            particle_filter.resample(weights)

        # ----------------------------------------------------
        # 4. Apply obstacle safety and update navigation.
        # The active waypoint is preserved while the robot slows/stops.
        # ----------------------------------------------------

        if state["obstacle_stop"]:
            if state["blocked_since"] is None:
                state["blocked_since"] = sim_time
        else:
            state["blocked_since"] = None

        controller.set_obstacle_safety(
            stop=state["obstacle_stop"],
            speed_scale=state["obstacle_speed_scale"],
        )
        controller.update(sim_time)

        # ----------------------------------------------------
        # 5. Get ground-truth, odometry and PF poses
        # ----------------------------------------------------

        # Ground truth comes directly from PyBullet.
        true_pos, true_orn = p.getBasePositionAndOrientation(
            robot.agv_id
        )

        true_theta = p.getEulerFromQuaternion(true_orn)[2]

        # Odometry estimate.
        odom_x, odom_y, odom_theta = odometry.get_pose()

        # Particle Filter estimate.
        pf_x, pf_y, pf_theta = state["pf_pose"]

        # ----------------------------------------------------
        # 6. Log all localization estimates to CSV
        # ----------------------------------------------------

        if localization_writer is not None:
            elapsed_real_s = time.perf_counter() - RUN_START_REAL
            localization_writer.writerow([
                SCENARIO_ID,
                sim_time,
                elapsed_real_s,
                robot_id,

                true_pos[0],
                true_pos[1],
                true_theta,

                odom_x,
                odom_y,
                odom_theta,

                pf_x,
                pf_y,
                pf_theta,
            ])

    # Flush periodically so the CSV is not held in memory
    # until the end of the simulation.
    if (
        localization_file is not None
        and clock["physics_steps"] % 240 == 0
    ):
        localization_file.flush()
        if obstacle_file is not None:
            obstacle_file.flush()


# Register the callback with the simulation.
world.add_step_callback(control_callback)


# --------------------------------------------------------------
# Execute both waypoint sequences concurrently
# --------------------------------------------------------------

try:
    with open(
        LOCALIZATION_CSV_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as log_file, open(
        OBSTACLE_CSV_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as event_file:

        localization_file = log_file
        localization_writer = csv.writer(log_file)
        obstacle_file = event_file
        obstacle_writer = csv.writer(event_file)

        obstacle_writer.writerow([
            "scenario_id",
            "sim_time_s",
            "elapsed_real_s",
            "robot_id",
            "safety_state",
            "tracked_candidates",
            "speed_scale",
        ])

        # Each row records the true pose and both localization estimates.
        localization_writer.writerow([
            "scenario_id",
            "sim_time_s",
            "elapsed_real_s",
            "robot_id",

            "true_x",
            "true_y",
            "true_theta",

            "odom_x",
            "odom_y",
            "odom_theta",

            "pf_x",
            "pf_y",
            "pf_theta",
        ])

        # ----------------------------------------------------
        # Start the first waypoint for each robot
        # ----------------------------------------------------

        for robot_id, state in robots.items():

            if state["waypoints"]:
                x_goal, y_goal = state["waypoints"][0]

                print(
                    f"\n{robot_id}: starting toward "
                    f"({x_goal:.2f}, {y_goal:.2f})"
                )

                state["controller"].start_go_to_position(
                    x_goal,
                    y_goal,
                    sim_time=0.0,
                    position_tolerance=POSITION_TOLERANCE,
                    angle_tolerance_deg=ANGLE_TOLERANCE_DEG,
                    max_attempts=MAX_ATTEMPTS,
                )

            else:
                state["finished"] = True

        # ----------------------------------------------------
        # Main simulation loop
        # ----------------------------------------------------

        while not all(
            state["finished"] or state["failed"]
            for state in robots.values()
        ):

            # Advance PyBullet by one physics step.
            world.simStep()

            sim_time = (
                clock["physics_steps"] * world.physics_dt
            )

            # Replan only after a continuous 1.5 s STOP condition.
            for robot_id, state in robots.items():
                controller = state["controller"]
                blocked_since = state["blocked_since"]
                if (
                    state["finished"] or state["failed"]
                    or blocked_since is None
                    or sim_time - blocked_since < BLOCKED_REPLAN_DELAY
                    or sim_time - state["last_replan_attempt"] < REPLAN_RETRY_INTERVAL
                    or controller.is_finished
                ):
                    continue

                state["last_replan_attempt"] = sim_time
                pose_x, pose_y, _ = controller.pose()
                start_cell = (int(math.floor(pose_x)), int(math.floor(pose_y)))
                goal_cell = (
                    int(math.floor(controller.x_goal)),
                    int(math.floor(controller.y_goal)),
                )

                new_grid_path = plan_around_tracks(
                    planner, start_cell, goal_cell, state["obstacle_tracker"].tracks
                )

                if not new_grid_path or len(new_grid_path) < 2:
                    print(
                        f"{robot_id}: no alternative path yet; "
                        f"will retry in {REPLAN_RETRY_INTERVAL:.1f} s."
                    )
                    continue

                compressed_path = compress_grid_path(new_grid_path)
                detour_waypoints = [
                    planner.cell_to_world(cell) for cell in compressed_path[1:]
                ]
                remaining_original = state["waypoints"][state["waypoint_index"] + 1:]
                state["waypoints"] = detour_waypoints + remaining_original
                state["waypoint_index"] = 0

                # Replace the remaining route visualization.
                for line_id in route_lines[robot_id]:
                    try:
                        p.removeUserDebugItem(line_id)
                    except Exception:
                        pass
                current_world = (pose_x, pose_y)
                route_world = [current_world] + state["waypoints"]
                color = ROBOT_COLORS.get(robot_id, [0.2, 0.8, 0.3])
                route_lines[robot_id] = draw_route(route_world, color)

                # Resume toward the first detour waypoint.
                state["obstacle_stop"] = False
                state["obstacle_speed_scale"] = 1.0
                state["blocked_since"] = None
                state["replan_count"] += 1
                next_x, next_y = state["waypoints"][0]
                controller.set_obstacle_safety(stop=False, speed_scale=1.0)
                controller.start_go_to_position(
                    next_x, next_y, sim_time=sim_time,
                    position_tolerance=POSITION_TOLERANCE,
                    angle_tolerance_deg=ANGLE_TOLERANCE_DEG,
                    max_attempts=MAX_ATTEMPTS,
                )
                print(
                    f"{robot_id}: replanned around dynamic obstacle "
                    f"after {BLOCKED_REPLAN_DELAY:.1f} s "
                    f"(replan #{state['replan_count']})."
                )

            # Keep Matplotlib responsive without updating it
            # on every physics step.
            if (
                clock["physics_steps"]
                % VISUALIZATION_INTERVAL == 0
            ):
                try:
                    plt.pause(0.001)
                except Exception:
                    pass

            # ------------------------------------------------
            # Check for waypoint completion for each robot
            # ------------------------------------------------

            for robot_id, state in robots.items():

                controller = state["controller"]

                if (
                    state["finished"]
                    or state["failed"]
                    or not controller.is_finished
                ):
                    continue

                # A finished controller may have succeeded or failed.
                if not controller.success:
                    print(
                        f"{robot_id}: navigation failed at "
                        f"waypoint {state['waypoint_index'] + 1}."
                    )

                    state["failed"] = True
                    controller.stop()
                    continue

                # Identify the waypoint just reached.
                reached_position = state["waypoints"][
                    state["waypoint_index"]
                ]

                reached_cell = world_to_cell(reached_position)

                # Remove completed pickup and delivery markers.
                update_completed_markers(
                    robot_id,
                    reached_cell,
                    job_markers,
                    matplotlib_job_artists,
                )

                # Remove the corresponding route line in PyBullet.
                line_index = state["waypoint_index"]
                lines = route_lines[robot_id]

                if line_index < len(lines):
                    p.removeUserDebugItem(lines[line_index])

                state["waypoint_index"] += 1

                # If no waypoints remain, this robot is done.
                if (
                    state["waypoint_index"]
                    >= len(state["waypoints"])
                ):
                    state["finished"] = True

                    x, y, theta = controller.pose()

                    print(
                        f"{robot_id}: route completed at "
                        f"x={x:.2f}, y={y:.2f}, "
                        f"theta={math.degrees(theta):.1f} deg"
                    )

                    continue

                # Otherwise, start navigation toward the next waypoint.
                x_goal, y_goal = state["waypoints"][
                    state["waypoint_index"]
                ]

                controller.start_go_to_position(
                    x_goal,
                    y_goal,
                    sim_time=sim_time,
                    position_tolerance=POSITION_TOLERANCE,
                    angle_tolerance_deg=ANGLE_TOLERANCE_DEG,
                    max_attempts=MAX_ATTEMPTS,
                )

    print("\nMulti-robot route execution finished.")
    print(f"Localization log saved to: {LOCALIZATION_CSV_FILE}")
    print("Press Ctrl+C to close the simulation.")

    # Keep PyBullet open for inspection after route execution.
    while True:
        world.simStep()

        if (
            clock["physics_steps"]
            % VISUALIZATION_INTERVAL == 0
        ):
            try:
                plt.pause(0.001)
            except Exception:
                pass

except KeyboardInterrupt:
    print("\nSimulation interrupted.")

finally:
    # Save the overall run summary, including when the user interrupts it.
    elapsed_real_s = time.perf_counter() - RUN_START_REAL
    sim_time_s = clock["physics_steps"] * PHYSICS_DT

    with open(SUMMARY_CSV_FILE, "w", newline="", encoding="utf-8") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow([
            "scenario_id", "map_file", "optimization_mode",
            "sim_time_s", "elapsed_real_s", "robot_id",
            "finished", "failed", "waypoints_reached", "waypoints_total",
        ])
        for robot_id, state in robots.items():
            summary_writer.writerow([
                SCENARIO_ID, str(MAP_FILE), OPTIMIZATION_MODE,
                sim_time_s, elapsed_real_s, robot_id,
                state["finished"], state["failed"],
                state["waypoint_index"], len(state["waypoints"]),
            ])

    print(f"Localization CSV: {LOCALIZATION_CSV_FILE}")
    print(f"Planning CSV: {PLAN_CSV_FILE}")
    print(f"Obstacle events CSV: {OBSTACLE_CSV_FILE}")
    print(f"Integrated summary CSV: {SUMMARY_CSV_FILE}")
    print(f"Route image: {ROUTE_IMAGE_FILE}")

    # Stop the robots before closing the simulation.
    for state in robots.values():
        state["controller"].stop()

    plt.close("all")
    world.end()

