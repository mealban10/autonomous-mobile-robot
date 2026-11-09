import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from utils.path_planner import AStarPlanner
from utils.task_manager import Job, TaskManager
from utils.fleet_manager import FleetManager, Robot


MAP_FILE = "maps/demoMap"

ROBOTS = [
    Robot("Robot 1", (1, 0)),
    Robot("Robot 2", (23, 17)),
]

# All locations are grid cells (column, row).
JOBS = [
    Job("Job A", pickup=(3, 2), delivery=(7, 5)),
    Job("Job B", pickup=(20, 15), delivery=(16, 12)),
    Job("Job C", pickup=(11, 14), delivery=(9, 9)),
    Job("Job D", pickup=(7, 9), delivery=(17, 9)),
]

# Available optimization modes:
# "movements" -> minimize total movements across all robots
# "time"      -> minimize the time until the last robot finishes
OPTIMIZATION_MODE = "time"

def plot_routes(planner, plans):
    grid = np.array(planner.grid)
    occupancy = (grid != 0).astype(int)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(
        occupancy,
        cmap=ListedColormap(["white", "dimgray"]),
        origin="upper",
        interpolation="nearest",
    )

    colors = ["dodgerblue", "darkorange", "mediumseagreen", "purple"]
    for index, plan in enumerate(plans):
        color = colors[index % len(colors)]
        path = plan["path"]
        path_x = [col for col, row in path]
        path_y = [row for col, row in path]
        ax.plot(path_x, path_y, color=color, linewidth=2.5,
                marker="o", markersize=2.5,
                label=f'{plan["robot_id"]}: {", ".join(plan["jobs"])}')

        for job_route in plan["job_routes"]:
            pickup = job_route["to_pickup"][-1]
            delivery = job_route["to_delivery"][-1]
            ax.scatter(*pickup, marker="s", s=100, color=color,
                       edgecolors="black", zorder=4)
            ax.scatter(*delivery, marker="*", s=180, color=color,
                       edgecolors="black", zorder=4)
            ax.annotate(job_route["job_id"], delivery, xytext=(4, 4),
                        textcoords="offset points", fontsize=8)

    for robot in ROBOTS:
        ax.scatter(*robot.start, marker="o", s=120, color="limegreen",
                   edgecolors="black", zorder=5)
        ax.annotate(robot.robot_id, robot.start, xytext=(5, 5),
                    textcoords="offset points", fontsize=8)

    ax.set_xticks(np.arange(-0.5, planner.cols, 1))
    ax.set_yticks(np.arange(-0.5, planner.rows, 1))
    ax.grid(color="lightgray", linewidth=0.5)
    ax.set_xlim(-0.5, planner.cols - 0.5)
    ax.set_ylim(planner.rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Multi-robot routes with automatic job assignment")
    ax.legend()
    plt.tight_layout()
#    plt.savefig("multi_robot_routes_time.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    planner = AStarPlanner(MAP_FILE)
    task_manager = TaskManager(JOBS)
    fleet_manager = FleetManager(planner, task_manager, ROBOTS)

    print(f"Map dimensions: {planner.cols} columns x {planner.rows} rows")
    print(f"Optimization mode: {OPTIMIZATION_MODE}")

    plans, best_score = fleet_manager.choose_best_assignment(task_manager.pending_jobs(),optimization_mode=OPTIMIZATION_MODE)

    fleet_manager.commit_assignment(plans)

    print("\nBest automatic assignment:")

    for plan in plans:
        print(f'  {plan["robot_id"]}: {", ".join(plan["jobs"])}')

        for job_route in plan["job_routes"]:
            print(f'{job_route["job_id"]}: {job_route["cost"]} movements')

    if OPTIMIZATION_MODE == "movements":
        print(f"Total movements: {best_score}")
    else:
        print(f"Estimated completion cost: {best_score} movements")

    print("\nJob states:")
    for job in task_manager.jobs:
        print(f"  {job.job_id}: {job.status.value}, robot={job.assigned_robot}")

    plot_routes(planner, plans)


if __name__ == "__main__":
    main()
