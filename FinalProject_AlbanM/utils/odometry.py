import numpy as np
import pybullet as p
import pygame
import os

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)


class OdometryVisualisation:

    #---------------------------------------------------------------------------------------------------
    # __init__()	Initializes Pygame, creates the window, and prepares the drawing surface.
    #---------------------------------------------------------------------------------------------------

    def __init__(self, odo, dimensions, robotID):
        # Visualization settings
        self.odo = odo
        self.robotID = robotID
        self.draw_boxes = []

        pygame.init()
        self.UNIT = 10

        self.displaysurface = pygame.display.set_mode((dimensions[0] * self.UNIT, dimensions[1] * self.UNIT))
        self.displaysurface.fill((0, 0, 0))

        pygame.display.flip()
        self.clock = pygame.time.Clock()

    #---------------------------------------------------------------------------------------------------
    # load_map(fname)	Reads a text-based map and stores the cells that should be drawn as obstacles.
    #---------------------------------------------------------------------------------------------------

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
                    self.draw_boxes.append((x * self.UNIT, y * self.UNIT, self.UNIT, self.UNIT))

                col += 1

            row += 1

    #---------------------------------------------------------------------------------------------------
    # redraw()	Clears the screen, draws the obstacles, and updates the window.
    #---------------------------------------------------------------------------------------------------

    def redraw(self):
        # Redraw the map
        self.displaysurface.fill((0, 0, 0))

        for i in self.draw_boxes:
            pygame.draw.rect(self.displaysurface, (255, 255, 255), i)

        # Get the true robot pose
        pos, orn = p.getBasePositionAndOrientation(self.robotID)
        true_theta = p.getEulerFromQuaternion(orn)[2]

        pygame.display.flip()
        self.clock.tick(20)

    #---------------------------------------------------------------------------------------------------
    # pygame_event_clean()	Processes pending Pygame events.
    #---------------------------------------------------------------------------------------------------

    def pygame_event_clean(self):
        pygame.event.pump()



class DifferentialDriveOdometry:

    def __init__(
        self,
        wheel_radius=0.12,
        wheel_base=0.56,
        left_wheel_joint=1,
        right_wheel_joint=0,
        noise_enabled=False,
        quantization_enabled=False,
        encoder_ticks_per_rev=360,
        wheel_angle_noise_std=0.001,
        random_seed=None,
    ):

        # Robot geometry.
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base

        self.left_joint = left_wheel_joint
        self.right_joint = right_wheel_joint

        # Sensor simulation settings.
        self.noise_enabled = noise_enabled
        self.quantization_enabled = quantization_enabled
        self.encoder_ticks_per_rev = encoder_ticks_per_rev
        self.wheel_angle_noise_std = wheel_angle_noise_std

        if encoder_ticks_per_rev <= 0:
            raise ValueError(
                "encoder_ticks_per_rev must be positive."
            )

        # Angular resolution of one encoder tick.
        self.encoder_tick_angle = (
            2.0 * np.pi / encoder_ticks_per_rev
        )

        # Independent random generator for sensor noise.
        self.rng = np.random.default_rng(random_seed)

        # Estimated robot pose.
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Previous encoder readings.
        self.prev_left_pos = None
        self.prev_right_pos = None

    # --------------------------------------------------------
    # Reset odometry
    # --------------------------------------------------------

    def reset(self, x=0.0, y=0.0, theta=0.0):

        self.x = x
        self.y = y
        self.theta = np.arctan2(
            np.sin(theta),
            np.cos(theta),
        )

        self.prev_left_pos = None
        self.prev_right_pos = None

    # --------------------------------------------------------
    # Simulate encoder measurements
    # --------------------------------------------------------

    def _quantize_wheel_angle(self, wheel_angle):
        """Quantize an absolute wheel angle to encoder ticks."""

        tick_count = np.rint(
            wheel_angle / self.encoder_tick_angle
        )

        return tick_count * self.encoder_tick_angle

    def _add_wheel_measurement_noise(self, delta_phi):
        """Add noise to a wheel's measured angular increment."""

        # Avoid inventing wheel motion when there is no measured
        # increment. This is a simple sensor-noise model.
        if abs(delta_phi) < 1e-12:
            return delta_phi

        noise = self.rng.normal(
            loc=0.0,
            scale=self.wheel_angle_noise_std,
        )

        return delta_phi + noise

    # --------------------------------------------------------
    # Update odometry
    # --------------------------------------------------------

    def update(self, robot_id):

        # Read the actual wheel joint angles from PyBullet.
        left_pos = p.getJointState(
            robot_id,
            self.left_joint,
        )[0]

        right_pos = p.getJointState(
            robot_id,
            self.right_joint,
        )[0]

        # First call: establish the encoder reference.
        if self.prev_left_pos is None:

            if self.quantization_enabled:
                left_measurement = self._quantize_wheel_angle(
                    left_pos
                )
                right_measurement = self._quantize_wheel_angle(
                    right_pos
                )
            else:
                left_measurement = left_pos
                right_measurement = right_pos

            self.prev_left_pos = left_measurement
            self.prev_right_pos = right_measurement

            return 0.0, 0.0

        # ----------------------------------------------------
        # 1. Simulate encoder quantization
        # ----------------------------------------------------

        if self.quantization_enabled:

            left_measurement = self._quantize_wheel_angle(
                left_pos
            )
            right_measurement = self._quantize_wheel_angle(
                right_pos
            )

        else:
            left_measurement = left_pos
            right_measurement = right_pos

        # Calculate increments from consecutive measurements.
        delta_phi_left = (
            left_measurement - self.prev_left_pos
        )

        delta_phi_right = (
            right_measurement - self.prev_right_pos
        )

        # Save the current readings for the next update.
        self.prev_left_pos = left_measurement
        self.prev_right_pos = right_measurement

        # ----------------------------------------------------
        # 2. Simulate encoder measurement noise
        # ----------------------------------------------------

        if self.noise_enabled:

            delta_phi_left = self._add_wheel_measurement_noise(
                delta_phi_left
            )

            delta_phi_right = self._add_wheel_measurement_noise(
                delta_phi_right
            )

        # ----------------------------------------------------
        # 3. Convert wheel rotation to linear displacement
        # ----------------------------------------------------

        # Negative wheel rotation corresponds to forward motion
        # in the current robot model.
        delta_s_left = (
            -self.wheel_radius * delta_phi_left
        )

        delta_s_right = (
            -self.wheel_radius * delta_phi_right
        )

        # Differential-drive kinematics.
        delta_s = (
            delta_s_right + delta_s_left
        ) / 2.0

        delta_theta = (
            self.wheel_radius
            * (delta_phi_right - delta_phi_left)
        ) / self.wheel_base

        # ----------------------------------------------------
        # 4. Integrate the estimated robot pose
        # ----------------------------------------------------

        theta_mid = self.theta + delta_theta / 2.0

        self.x += delta_s * np.cos(theta_mid)
        self.y += delta_s * np.sin(theta_mid)
        self.theta += delta_theta

        # Normalize heading to [-pi, pi].
        self.theta = np.arctan2(
            np.sin(self.theta),
            np.cos(self.theta),
        )

        return delta_s, delta_theta

    # --------------------------------------------------------
    # Retrieve estimated pose
    # --------------------------------------------------------

    def get_pose(self):
        return self.x, self.y, self.theta

    # --------------------------------------------------------
    # Get 2D transformation matrix
    # --------------------------------------------------------

    def get_pose_2d_transform(self):

        c = np.cos(self.theta)
        s = np.sin(self.theta)

        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0,  1],
        ])
