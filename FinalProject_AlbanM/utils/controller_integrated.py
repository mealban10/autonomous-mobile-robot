import math
import pybullet as p

def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

class MotionController:

    #---------------------------------------------------------------------------------------------------
    # __init__()    Define initial configuration
    #---------------------------------------------------------------------------------------------------

    def __init__(
        self,
        robot,
        world,
        odometry,
        drive_speed=1.5,
        turn_speed=0.8,
        motor_force=50,
        position_tolerance=0.05,
        angle_tolerance_deg=2.0,
        timeout=60.0,
        localization_mode="odometry",
    ):

        # References to the existing robot, simulation world, and odometry objects.
        self.robot = robot
        self.world = world
        self.odo = odometry

        # Configured maximum speeds for linear movement and turning.
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed

        # Stored motor force limit; it only has an effect if used by the controller.
        self.motor_force = motor_force

        # Allowed position error in meters.
        self.position_tolerance = position_tolerance

        # Allowed angular error for completing a turn.
        self.angle_tolerance = math.radians(angle_tolerance_deg)

        # Internal threshold to select the navigation mode.
        # Large heading errors turn in place; smaller errors are corrected while moving.
        self.turn_threshold = math.radians(80.0)

        # Maximum time allowed for a movement operation, in seconds.
        self.timeout = timeout

        # Proportional gain used to calculate the turning correction.
        self.kp_angle = 10

        # Gain used to turn back toward the path when the robot drifts sideways.
        # This is only used for waypoint navigation, not direct distance moves.
        self.kp_cross_track = 25

        # Limit the extra heading angle introduced by lateral path error.
        # Keeping this bounded prevents abrupt steering and excessive zigzagging.
        self.max_cross_track_angle = math.radians(80.0)

        # Maximum wheel-speed difference command used for steering while moving.
        self.max_correction = 500

        # Minimum commanded wheel-speed magnitude used when turning.
        self.min_turn_speed = 0.4

        # Minimum commanded wheel-speed magnitude used when driving.
        self.min_drive_speed = 0.45

        # Current movement state.
        self.phase = "idle"
        self.movement_type = None
        self.start_time = 0.0
        self.last_sim_time = 0.0
        self.is_finished = True
        self.success = False

        # Targets and movement progress.
        self.target_theta = None
        self.start_theta = None
        self.start_x = None
        self.start_y = None
        self.target_distance = 0.0
        self.direction = 1.0
        self.x_goal = None
        self.y_goal = None
        self.position_tol = position_tolerance
        self.angle_tol = self.angle_tolerance
        self.max_attempts = 5
        self.attempts = 0
        # Latch reverse mode after overshooting a waypoint.
        self.reverse_to_waypoint = False

        # Last wheel commands, used for diagnostics.
        self.left_command = 0.0
        self.right_command = 0.0

        # External obstacle-safety override, updated by the main loop.
        self.obstacle_stop = False
        self.obstacle_speed_scale = 1.0

        # Select the pose estimate used by the navigation controller.
        if localization_mode not in ("odometry", "particle_filter"):
            raise ValueError(
                "localization_mode must be 'odometry' or 'particle_filter'"
            )

        self.localization_mode = localization_mode
        self.pf_pose = self.odo.get_pose()

        self.obstacle_stop = False
        self.obstacle_speed_scale = 1.0

    #---------------------------------------------------------------------------------------------------
    # pose()    Returns the robot's estimated pose: x, y, and theta.
    #---------------------------------------------------------------------------------------------------

    def pose(self):
        # Return the pose estimate selected for navigation.
        if self.localization_mode == "particle_filter":
            return self.pf_pose

        return self.odo.get_pose()

    #---------------------------------------------------------------------------------------------------
    # set_particle_pose()    Get particle filter pose
    #---------------------------------------------------------------------------------------------------

    def set_particle_pose(self, x, y, theta):
        # Store the latest particle-filter pose estimate.
        self.pf_pose = (x, y, wrap_angle(theta))

    #---------------------------------------------------------------------------------------------------
    # command_wheels(left, right)    Sends the desired speeds to the left and right wheels.
    #---------------------------------------------------------------------------------------------------

    def command_wheels(self, left, right):
        self.left_command = left
        self.right_command = right
        self.robot.set_wheel_speeds(left, right)

    #---------------------------------------------------------------------------------------------------
    # stop()    Sends a target speed of zero to both wheels.
    #---------------------------------------------------------------------------------------------------

    def set_obstacle_safety(self, stop=False, speed_scale=1.0):
        """Apply a safety override without changing the active waypoint."""
        self.obstacle_stop = bool(stop)
        self.obstacle_speed_scale = max(0.0, min(1.0, float(speed_scale)))

    def stop(self):
        self.command_wheels(0.0, 0.0)

    #---------------------------------------------------------------------------------------------------
    # _clip(value, low, high)    Constrains a value to remain within a specified range.
    #---------------------------------------------------------------------------------------------------

    def _clip(self, value, low, high):
        return max(low, min(high, value))

    #---------------------------------------------------------------------------------------------------
    # _finish(success)    Finishes the current movement without advancing the simulation.
    #---------------------------------------------------------------------------------------------------

    def _finish(self, success):
        self.stop()
        self.is_finished = True
        self.success = success
        self.phase = "idle"
        self.movement_type = None

    #---------------------------------------------------------------------------------------------------
    # start_turn(angle_deg)    Starts a turn without blocking the main program.
    #---------------------------------------------------------------------------------------------------

    def start_turn(self, angle_deg, sim_time=0.0):
        self.stop()
        _, _, theta = self.pose()

        self.movement_type = "turn"
        self.phase = "turn"
        self.start_theta = theta
        self.target_theta = wrap_angle(theta + math.radians(angle_deg))
        self.start_time = sim_time
        self.last_sim_time = sim_time
        self.is_finished = False
        self.success = False

    #---------------------------------------------------------------------------------------------------
    # start_move(distance_m)    Starts a linear movement without blocking the main program.
    #---------------------------------------------------------------------------------------------------

    def start_move(self, distance_m, sim_time=0.0):
        self.stop()

        x, y, theta = self.pose()

        self.start_time = sim_time
        self.last_sim_time = sim_time
        self.is_finished = False
        self.success = False

        if abs(distance_m) <= self.position_tolerance:
            self._finish(True)
            return

        self.movement_type = "move"
        self.phase = "move"
        self.start_x = x
        self.start_y = y
        self.start_theta = theta

        # Keep the initial heading as the desired heading throughout this straight segment.
        self.target_theta = theta

        self.target_distance = abs(distance_m)
        self.direction = 1.0 if distance_m > 0 else -1.0

    #---------------------------------------------------------------------------------------------------
    # start_go_to_position(...)    Starts navigation to a waypoint without blocking.
    #---------------------------------------------------------------------------------------------------

    def start_go_to_position(
        self,
        x_goal,
        y_goal,
        sim_time=0.0,
        position_tolerance=None,
        angle_tolerance_deg=None,
        max_attempts=5,
    ):
        self.stop()

        self.x_goal = x_goal
        self.y_goal = y_goal
        self.position_tol =  self.position_tolerance  if position_tolerance is None else position_tolerance
        self.angle_tol = self.angle_tolerance if angle_tolerance_deg is None else math.radians(angle_tolerance_deg)
        self.max_attempts = max_attempts
        self.attempts = 0

        self.movement_type = "waypoint"
        self.start_time = sim_time
        self.last_sim_time = sim_time
        self.is_finished = False
        self.success = False

        self._prepare_waypoint(sim_time)

    #---------------------------------------------------------------------------------------------------
    # _prepare_waypoint(...)    Calculates the next turn or translation toward the waypoint.
    #---------------------------------------------------------------------------------------------------

    def _prepare_waypoint(self, sim_time):
        x, y, theta = self.pose()
        dx = self.x_goal - x
        dy = self.y_goal - y
        distance = math.hypot(dx, dy)

        if distance <= self.position_tol:
            self._finish(True)
            return

        if self.attempts >= self.max_attempts:
            self._finish(False)
            return

        self.attempts += 1

        # Calculate the desired heading toward the waypoint.
        target_theta = math.atan2(dy, dx)
        angle_error = wrap_angle(target_theta - theta)

        self.target_theta = wrap_angle(target_theta)
        self.target_distance = distance
        self.start_x = x
        self.start_y = y
        self.start_theta = theta
        self.direction = 1.0
        self.reverse_to_waypoint = False

        # Select the motion mode using the internal 45-degree threshold.
        # angle_tol remains the tolerance for completing the turn.
        if abs(angle_error) > self.turn_threshold:
            self.phase = "turn"
        else:
            self.phase = "move"

        self.last_sim_time = sim_time

    #------------------------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------------------------

    def set_obstacle_safety(self, stop=False, speed_scale=1.0):
        self.obstacle_stop = bool(stop)
        self.obstacle_speed_scale = max(0.0, min(1.0, speed_scale))

    #---------------------------------------------------------------------------------------------------
    # update(sim_time)    Updates the active movement using the current simulation time.
    #---------------------------------------------------------------------------------------------------


    def update(self, sim_time):
        if self.is_finished:
            return

        self.last_sim_time = sim_time

        # Use simulation time rather than wall-clock time for movement timeouts.
        if sim_time - self.start_time >= self.timeout:
            self._finish(False)
            return

        x, y, theta = self.pose()

        # TURN: rotate in place until the desired heading is reached.
        if self.phase == "turn":
            error = wrap_angle(self.target_theta - theta)

            if abs(error) <= self.angle_tol:
                self.stop()

                if self.movement_type == "waypoint":
                    dx = self.x_goal - x
                    dy = self.y_goal - y
                    distance = math.hypot(dx, dy)

                    if distance <= self.position_tol:
                        self._finish(True)
                        return

                    self.start_x = x
                    self.start_y = y
                    self.start_theta = self.target_theta
                    self.target_distance = distance
                    self.direction = 1.0
                    self.phase = "move"
                else:
                    self._finish(True)

                return

            # Positive angular velocity means a counterclockwise turn.
            omega = self._clip(
                self.kp_angle * error,
                -self.turn_speed,
                self.turn_speed
            )

            if abs(omega) < self.min_turn_speed:
                omega = math.copysign(self.min_turn_speed, error)

            # Wheel-speed convention: negative equal speeds move forward.
            left = -omega
            right = omega
            self.command_wheels(left, right)
            return

        # MOVE: travel along a straight segment while correcting heading.
        if self.phase == "move":

            if self.obstacle_stop:
                self.stop()
                return

            dx = x - self.start_x
            dy = y - self.start_y

            # Calculate signed progress
            progress = (
                dx * math.cos(self.start_theta)
                + dy * math.sin(self.start_theta)
            ) * self.direction

            distance_remaining = self.target_distance - progress

            # Calculate waypoint completion
            if self.movement_type == "waypoint":
                distance_to_goal = math.hypot(
                    self.x_goal - x,
                    self.y_goal - y
                )

                # Distance to the waypoint
                distance_to_goal = math.hypot(
                    self.x_goal - x,
                    self.y_goal - y
                )

                # Allow a small longitudinal overshoot.
                overshoot_tolerance = max(self.position_tol, 0.15)

                segment_complete = (
                    distance_to_goal <= self.position_tol
                    or (
                        distance_remaining <= 0.0
                        and distance_to_goal <= overshoot_tolerance
                    )
                )

                distance_for_speed = distance_to_goal

            else:
                segment_complete = distance_remaining <= self.position_tol
                distance_for_speed = max(0.0, distance_remaining)

            if segment_complete:
                self.stop()

                if self.movement_type == "waypoint":
                    self._prepare_waypoint(sim_time)
                else:
                    self._finish(True)

                return

            # Reduce linear speed as the target approaches.
            speed = min(
                self.drive_speed,
                max(
                    self.min_drive_speed,
                    math.sqrt(
                        max(0.0, 2.0 * self.drive_speed * distance_for_speed)
                    )
                )
            )

            speed *= self.obstacle_speed_scale

            # Keep a fixed heading throughout the straight segment.
            # The next waypoint defines a new heading after this segment ends.
            desired_heading = self.target_theta
            heading_error = wrap_angle(desired_heading - theta)

            # Limit steering correction so neither wheel reverses while driving.
            # Negative wheel speeds represent forward motion in this robot.
            max_steering = 0.35 * speed
            correction = self._clip(
                self.kp_angle * heading_error,
                -max_steering,
                max_steering
            )

            # Negative wheel speeds move forward.
            forward = -self.direction * speed

            left = forward - correction
            right = forward + correction

            self.command_wheels(left, right)
            return


    #---------------------------------------------------------------------------------------------------
    # go_to_position(...)    Compatibility wrapper for starting waypoint navigation.
    #---------------------------------------------------------------------------------------------------

    def go_to_position(
        self,
        x_goal,
        y_goal,
        position_tolerance=None,
        angle_tolerance_deg=None,
        max_attempts=5,
        sim_time=0.0,
    ):
        self.start_go_to_position(
            x_goal,
            y_goal,
            sim_time=sim_time,
            position_tolerance=position_tolerance,
            angle_tolerance_deg=angle_tolerance_deg,
            max_attempts=max_attempts,
        )

    #---------------------------------------------------------------------------------------------------
    # turn_angle(angle_deg)    Compatibility wrapper for starting a turn.
    #---------------------------------------------------------------------------------------------------

    def turn_angle(self, angle_deg, sim_time=0.0):
        self.start_turn(angle_deg, sim_time=sim_time)

    #---------------------------------------------------------------------------------------------------
    # move_distance(distance_m)    Compatibility wrapper for starting a linear movement.
    #---------------------------------------------------------------------------------------------------

    def move_distance(self, distance_m, sim_time=0.0):
        self.start_move(distance_m, sim_time=sim_time)
