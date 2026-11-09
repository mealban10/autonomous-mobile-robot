from dataclasses import dataclass, field
from itertools import permutations

from utils.task_manager import Job, TaskManager
from utils.path_planner import AStarPlanner


#---------------------------------------------------------------------------------------------------
# Robot    Stores the robot's initial position and assigned jobs.
#---------------------------------------------------------------------------------------------------
@dataclass
class Robot:
    robot_id: str
    start: tuple[int, int]
    assigned_jobs: list[str] = field(default_factory=list)


class FleetManager:

    #---------------------------------------------------------------------------------------------------
    # __init__(...)    Initializes the fleet manager and registers the available robots.
    #---------------------------------------------------------------------------------------------------
    def __init__(self, planner: AStarPlanner, task_manager: TaskManager,
                 robots: list[Robot]):
        self.planner = planner
        self.task_manager = task_manager
        self.robots = {robot.robot_id: robot for robot in robots}
        if len(self.robots) != len(robots):
            raise ValueError("Robot IDs must be unique.")

    #---------------------------------------------------------------------------------------------------
    # plan_job_from(...)    Plans the route from a given position to a job's pickup and delivery locations.
    #---------------------------------------------------------------------------------------------------
    def plan_job_from(self, start: tuple[int, int], job: Job) -> dict:
        """Plan from a current cell to pickup and then delivery."""
        to_pickup = self.planner.plan(start, job.pickup)
        if to_pickup is None:
            return {"to_pickup": None, "to_delivery": None, "path": None}

        to_delivery = self.planner.plan(job.pickup, job.delivery)
        if to_delivery is None:
            return {"to_pickup": to_pickup, "to_delivery": None, "path": None}

        return {
            "to_pickup": to_pickup,
            "to_delivery": to_delivery,
            "path": to_pickup + to_delivery[1:],
        }

    #---------------------------------------------------------------------------------------------------
    # plan_job(...)    Plans the complete route for one job from the robot's initial position.
    #---------------------------------------------------------------------------------------------------
    def plan_job(self, robot_id: str, job: Job) -> dict:
        """Plan a job from the robot's initial cell."""
        route = self.plan_job_from(self.robots[robot_id].start, job)
        return {
            "robot_id": robot_id,
            "job_id": job.job_id,
            **route,
        }

    #---------------------------------------------------------------------------------------------------
    # plan_sequence(...)    Plans an ordered sequence of jobs and calculates the total route cost.
    #---------------------------------------------------------------------------------------------------
    def plan_sequence(self, robot_id: str, jobs: list[Job]) -> dict:
        current = self.robots[robot_id].start
        full_path = []
        job_routes = []
        total_cost = 0

        for job in jobs:
            route = self.plan_job_from(current, job)
            if route["path"] is None:
                return {
                    "robot_id": robot_id, "jobs": [j.job_id for j in jobs],
                    "job_routes": [], "path": None, "cost": float("inf"),
                }

            # Join job routes without duplicating shared endpoints.
            if full_path:
                full_path.extend(route["path"][1:])
            else:
                full_path.extend(route["path"])

            job_cost = len(route["path"]) - 1
            total_cost += job_cost
            job_routes.append({
                "job_id": job.job_id,
                "to_pickup": route["to_pickup"],
                "to_delivery": route["to_delivery"],
                "path": route["path"],
                "cost": job_cost,
            })
            current = job.delivery

        return {
            "robot_id": robot_id,
            "jobs": [job.job_id for job in jobs],
            "job_routes": job_routes,
            "path": full_path,
            "cost": total_cost,
        }

    #---------------------------------------------------------------------------------------------------
    # choose_best_assignment(...)    Tests job assignments and sequences to find the lowest-cost solution.
    #---------------------------------------------------------------------------------------------------
    def choose_best_assignment(self, jobs: list[Job], optimization_mode: str = "time") -> tuple[list[dict], int]:

        from itertools import permutations, product

        if optimization_mode not in ("movements", "time"):
            raise ValueError(
                "optimization_mode must be 'movements' or 'time'."
            )

        robot_ids = list(self.robots)

        if not robot_ids:
            raise ValueError("At least one robot is required.")

        if not jobs:
            return [], 0

        best_plans = None
        best_score = float("inf")

        # Assign each job to a robot, then test job orders.
        for job_order in permutations(jobs):
            for robot_choices in product(robot_ids, repeat=len(jobs)):
                groups = {robot_id: [] for robot_id in robot_ids}

                for job, robot_id in zip(job_order, robot_choices):
                    groups[robot_id].append(job)

                plans = []
                total_movements = 0
                valid = True

                # Generate and validate the route assigned to each robot.
                for robot_id in robot_ids:
                    sequence = groups[robot_id]

                    if not sequence:
                        continue

                    plan = self.plan_sequence(robot_id, sequence)

                    if plan["path"] is None:
                        valid = False
                        break

                    plans.append(plan)
                    total_movements += plan["cost"]

                if not valid:
                    continue

                # Calculate the assignment score according to the selected optimization mode.
                if optimization_mode == "movements":
                    score = total_movements
                else:
                    # Estimate completion time using movement count.
                    score = max(
                        (plan["cost"] for plan in plans),
                        default=0,
                    )

                if score < best_score:
                    best_score = score
                    best_plans = plans

        if best_plans is None:
            raise ValueError("No valid assignment was found.")

        return best_plans, int(best_score)

    #---------------------------------------------------------------------------------------------------
    # commit_assignment(...)    Saves the selected job assignments in the fleet and task managers.
    #---------------------------------------------------------------------------------------------------
    def commit_assignment(self, plans: list[dict]) -> None:
        # Clear previous assignments before registering the new ones.
        for robot in self.robots.values():
            robot.assigned_jobs.clear()

        for plan in plans:
            robot_id = plan["robot_id"]
            for job_id in plan["jobs"]:
                self.task_manager.mark_assigned(job_id, robot_id)
                self.robots[robot_id].assigned_jobs.append(job_id)