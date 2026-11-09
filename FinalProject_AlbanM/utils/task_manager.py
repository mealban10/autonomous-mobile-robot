from dataclasses import dataclass
from enum import Enum


#---------------------------------------------------------------------------------------------------
# TaskStatus    Defines the possible states of a job.
#---------------------------------------------------------------------------------------------------
class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"


#---------------------------------------------------------------------------------------------------
# Job    Stores the information and current status of a material handling job.
#---------------------------------------------------------------------------------------------------
@dataclass
class Job:
    job_id: str
    pickup: tuple[int, int]
    delivery: tuple[int, int]
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: str | None = None


class TaskManager:

    #---------------------------------------------------------------------------------------------------
    # __init__(jobs)    Initializes the task manager with an optional list of jobs.
    #---------------------------------------------------------------------------------------------------
    def __init__(self, jobs: list[Job] | None = None):
        self.jobs = jobs or []

    #---------------------------------------------------------------------------------------------------
    # add_job(job)    Adds a new job, preventing duplicate job identifiers.
    #---------------------------------------------------------------------------------------------------
    def add_job(self, job: Job) -> None:
        if any(existing.job_id == job.job_id for existing in self.jobs):
            raise ValueError(f"Duplicate job id: {job.job_id}")
        self.jobs.append(job)

    #---------------------------------------------------------------------------------------------------
    # pending_jobs()    Returns all jobs that are currently pending.
    #---------------------------------------------------------------------------------------------------
    def pending_jobs(self) -> list[Job]:
        return [job for job in self.jobs if job.status == TaskStatus.PENDING]

    #---------------------------------------------------------------------------------------------------
    # mark_assigned(job_id, robot_id)    Assigns a pending job to a robot.
    #---------------------------------------------------------------------------------------------------
    def mark_assigned(self, job_id: str, robot_id: str) -> None:
        job = self._get_job(job_id)
        if job.status != TaskStatus.PENDING:
            raise ValueError(f"Job {job_id} is not pending.")
        job.status = TaskStatus.ASSIGNED
        job.assigned_robot = robot_id

    #---------------------------------------------------------------------------------------------------
    # mark_completed(job_id)    Marks an assigned job as completed.
    #---------------------------------------------------------------------------------------------------
    def mark_completed(self, job_id: str) -> None:
        job = self._get_job(job_id)
        if job.status != TaskStatus.ASSIGNED:
            raise ValueError(f"Job {job_id} is not assigned.")
        job.status = TaskStatus.COMPLETED

    #---------------------------------------------------------------------------------------------------
    # _get_job(job_id)    Retrieves a job by its identifier or raises an error if not found.
    #---------------------------------------------------------------------------------------------------
    def _get_job(self, job_id: str) -> Job:
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        raise KeyError(f"Unknown job id: {job_id}")