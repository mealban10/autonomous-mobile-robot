import numpy as np
import pybullet as p
import pygame
import os

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)


class OdometryVisualisation:
    def __init__(self, odo, dimensions, robotID):
        # Visualization settings
        self.odo = odo
        self.robotID = robotID
        self.draw_boxes = []

        pygame.init()
        self.UNIT = 10

        self.displaysurface = pygame.display.set_mode(
            (dimensions[0] * self.UNIT, dimensions[1] * self.UNIT)
        )
        self.displaysurface.fill((0, 0, 0))

        pygame.display.flip()
        self.clock = pygame.time.Clock()

    def load_map(self, fname):
        # Load map for visualization
        xs = open(fname, "r").read().strip()
        row = 0

        for l in xs.split("\n"):
            col = 0

            for c in l.strip():

                if c == "1" or c == "2":
                    x = col
                    y = row

                    self.draw_boxes.append(
                        (x * self.UNIT, y * self.UNIT, self.UNIT, self.UNIT)
                    )

                col += 1

            row += 1

    def redraw(self):
        # Redraw the map
        self.displaysurface.fill((0, 0, 0))

        for i in self.draw_boxes:
            pygame.draw.rect(
                self.displaysurface,
                (255, 255, 255),
                i
            )

        # Get the true robot pose
        pos, orn = p.getBasePositionAndOrientation(self.robotID)
        true_theta = p.getEulerFromQuaternion(orn)[2]

        pygame.display.flip()
        self.clock.tick(20)

    def pygame_event_clean(self):
        pygame.event.pump()


"""Created by Claude.AI"""


class DifferentialDriveOdometry:
    def __init__(
        self,
        wheel_radius,
        wheel_base,
        left_wheel_joints,
        right_wheel_joints
    ):
        # Odometry parameters
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.left_joints = left_wheel_joints
        self.right_joints = right_wheel_joints

        # Current estimated pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Previous wheel positions
        self.prev_left_pos = None
        self.prev_right_pos = None

    def reset(self, x=0.0, y=0.0, theta=0.0):
        # Reset odometry state
        self.x = x
        self.y = y
        self.theta = theta
        self.prev_left_pos = None
        self.prev_right_pos = None

    def update(self, robot_id):
        # Read current wheel positions
        left_wheels = [
            p.getJointState(robot_id, j)[0]
            for j in self.left_joints
        ]

        right_wheels = [
            p.getJointState(robot_id, j)[0]
            for j in self.right_joints
        ]

        left_pos = np.mean(left_wheels)
        right_pos = np.mean(right_wheels)

        # Initialize wheel positions
        if self.prev_left_pos is None:
            self.prev_left_pos = left_pos
            self.prev_right_pos = right_pos
            return

        # Calculate wheel displacement
        delta_left = left_pos - self.prev_left_pos
        delta_right = right_pos - self.prev_right_pos

        # Convert wheel rotation to distance
        left_distance = delta_left * self.wheel_radius
        right_distance = delta_right * self.wheel_radius

        # Calculate robot displacement and rotation
        delta_distance = (left_distance + right_distance) / 2.0
        delta_theta = (right_distance - left_distance) / self.wheel_base

        # Detect in-place turning
        turning_in_place = left_distance * right_distance < 0

        if turning_in_place:

            # Keep position fixed during rotation
            delta_x = 0.0
            delta_y = 0.0

            # Apply turning compensation
            turn_factor = 0.22
            delta_theta = delta_theta * turn_factor

        else:

            # Apply differential-drive kinematics
            if abs(delta_theta) < 1e-6:

                delta_x = delta_distance * np.cos(self.theta)
                delta_y = delta_distance * np.sin(self.theta)

                delta_theta = 0

            else:

                radius = delta_distance / delta_theta

                delta_x = radius * (
                    np.sin(self.theta + delta_theta)
                    - np.sin(self.theta)
                )

                delta_y = -radius * (
                    np.cos(self.theta + delta_theta)
                    - np.cos(self.theta)
                )

        # Update estimated position
        self.x -= delta_x
        self.y -= delta_y

        # Update estimated orientation
        self.theta += delta_theta

        # Normalize angle to [-pi, pi]
        self.theta = np.arctan2(
            np.sin(self.theta),
            np.cos(self.theta)
        )

        # Store current wheel positions
        self.prev_left_pos = left_pos
        self.prev_right_pos = right_pos

        forward_distance = -delta_distance

        return forward_distance, delta_theta

    def get_pose(self):
        # Return current pose
        return self.x, self.y, self.theta

    def get_pose_2d_transform(self):
        # Return pose as a 2D transformation matrix
        c = np.cos(self.theta)
        s = np.sin(self.theta)

        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0,  1]
        ])


def debug_text(odo, r, wld):
    # Get estimated pose
    x, y, theta = odo.get_pose()

    # Get ground-truth pose
    pos, orn = p.getBasePositionAndOrientation(r.agv_id)
    true_theta = p.getEulerFromQuaternion(orn)[2]

    return theta, true_theta