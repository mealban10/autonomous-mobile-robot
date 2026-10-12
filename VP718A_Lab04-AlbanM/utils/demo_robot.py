from turtle import distance

import pybullet as p
import numpy as np

class DemoRobot:
    def __init__(self, x=0, y=0):

        # Dimensions of the robot chassis
        body_length = 0.6
        body_width = 0.35
        body_height = 0.28

        # Wheel dimensions
        wheel_radius = 0.12
        wheel_width = 0.08

        # Create collision and visual shapes for the robot chassis
        chassis_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[body_length/2, body_width/2, body_height/2]
        )

        chassis_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[body_length/2, body_width/2, body_height/2],
            rgbaColor=[0.2, 0.4, 0.8, 1.0]
        )

        # Create collision and visual shapes for the four wheels
        wheel_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            height=wheel_width
        )

        wheel_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            length=wheel_width,
            rgbaColor=[0.3, 0.3, 0.3, 1.0]
        )

        # Wheel positions relative to the center of the chassis.
        # Wheels on the +Y side are the right wheels,
        # while wheels on the -Y side are the left wheels.
        wheel_positions = [
            [body_length/3,  body_width/2 + wheel_width/2, -wheel_width/2],  # Front right
            [body_length/3, -body_width/2 - wheel_width/2, -wheel_width/2],  # Front left
            [-body_length/3,  body_width/2 + wheel_width/2, -wheel_width/2], # Rear right
            [-body_length/3, -body_width/2 - wheel_width/2, -wheel_width/2]  # Rear left
        ]

        # Create the four wheels as links attached to the robot chassis.
        link_masses = [2.0] * 4  # Each wheel has a mass of 2 kg
        link_collision_shapes = [wheel_collision] * 4
        link_visual_shapes = [wheel_visual] * 4
        link_positions = wheel_positions

        # Rotate the cylindrical wheels so that their rotation axis
        # is aligned with the robot's lateral direction.
        link_orientations = [
            p.getQuaternionFromEuler([1.57, 0, 0])
        ] * 4

        # Inertial frames are centered and have no additional rotation.
        link_inertial_frame_positions = [[0, 0, 0]] * 4
        link_inertial_frame_orientations = [[0, 0, 0, 1]] * 4

        # All four wheels are attached directly to the main chassis.
        link_parent_indices = [0] * 4

        # Revolute joints allow the wheels to rotate.
        link_joint_types = [p.JOINT_REVOLUTE] * 4

        # Wheel rotation is defined around the Z axis of the joint.
        link_joint_axis = [[0, 0, 1]] * 4

        # Create the complete robot as one multibody.
        self.agv_id = p.createMultiBody(
            baseMass=50,  # Mass of the main chassis
            baseCollisionShapeIndex=chassis_collision,
            baseVisualShapeIndex=chassis_visual,
            basePosition=[x, y, wheel_radius + body_height/2],  # -0.02 Slightly above the ground

            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frame_positions,
            linkInertialFrameOrientations=link_inertial_frame_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axis
        )

        # Get the number of joints and store their indices.
        self.num_joints = p.getNumJoints(self.agv_id)
        self.wheel_joints = list(range(self.num_joints))

        # Configure the wheel motors and contact friction.
        for joint_id in self.wheel_joints:

            # Start with the wheels stopped.
            p.setJointMotorControl2(
                bodyUniqueId=self.agv_id,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

            # Lateral friction provides traction against the floor.
            # Rolling friction is kept small to allow the wheels to move.
            p.changeDynamics(
                self.agv_id,
                joint_id,
                lateralFriction=1.0,
                rollingFriction=0.01
            )

        # Initial wheel speeds.
        self.left_speed = 0
        self.right_speed = 0

        # Current automatic movement and remaining simulation steps
        self.action = None
        self.action_steps = 0

        # Pause between consecutive movements
        self.pause_steps = 0

        # LiDAR configuration
        self.num_lidar_beams = 21
        self.lidar_max_range = 10
        self.lidar_fov_min = -1.0
        self.lidar_fov_max = 1.0

        self.lidar_counter = 0

        self.person_ids = []
        self.person_detected = False

    def move_forward(self, distance):

        # Calibrated at the current forward speed: simulation steps per meter.
        steps_per_meter = 215

        self.action = "forward"
        self.action_steps = int(distance * steps_per_meter)

    def move_backward(self, distance):

        # Use the same distance calibration as forward movement.
        steps_per_meter = 215

        self.action = "backward"
        self.action_steps = int(distance * steps_per_meter)

    def turn_left(self, angle):

        #simulation steps for a 90-degree turn.
        steps_per_90 = 370

        self.action = "left"
        self.action_steps = int((angle / 90) * steps_per_90)

    def turn_right(self, angle):

        #simulation steps for a 90-degree turn.
        steps_per_90 = 370

        self.action = "right"
        self.action_steps = int((angle / 90) * steps_per_90)

    def wait(self, duration):
        self.action = "wait"
        self.action_steps = int(duration * 240)

    def ray_cast_lidar(self):

        # Get robot position and orientation
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.agv_id)

        # Extract the robot's current yaw angle
        true_theta = p.getEulerFromQuaternion(robot_orn)[2]

        # Generate the LiDAR beam angles relative to the robot
        angles = np.linspace(
            true_theta + self.lidar_fov_min,
            true_theta + self.lidar_fov_max,
            self.num_lidar_beams
        )

        ray_from = []
        ray_to = []

        for angle in angles:

            direction = np.array([
                np.cos(angle),
                np.sin(angle),
                0
            ])

            lidar_height = 0.20

            ray_start = np.array(robot_pos)
            ray_start[2] += lidar_height

            ray_from.append(ray_start)
            ray_to.append(
                np.array(robot_pos) + direction * self.lidar_max_range
            )

        # Cast all rays at once
        results = p.rayTestBatch(ray_from, ray_to)

        distances = []
        object_ids = []
        person_distance = None

        for result in results:
            hit_object_id = result[0]

            hit_fraction = result[2]

            if hit_object_id >= 0:
                distance = hit_fraction * self.lidar_max_range
            else:
                distance = self.lidar_max_range

            distances.append(distance)
            object_ids.append(hit_object_id)

            if hit_object_id in self.person_ids:
                if person_distance is None or distance < person_distance:
                    person_distance = distance

        return np.array(distances), angles, object_ids, person_distance

    def set_person_ids(self, person_ids):
        self.person_ids = person_ids

    def set_route(self, route):

        self.route = route
        self.route_index = 0

    def start_next_action(self):

        if self.route_index >= len(self.route):
            return

        action, value = self.route[self.route_index]

        if action == "forward":
            self.move_forward(value)

        elif action == "backward":
            self.move_backward(value)

        elif action == "left":
            self.turn_left(value)

        elif action == "right":
            self.turn_right(value)

        elif action == "wait":
            self.wait(value)

        self.route_index += 1  

    

    def user_control(self, keys):

        # Stop the robot unless a movement key is being pressed.
        self.left_speed = 0
        self.right_speed = 0

        # Forward and backward movement
        if ord('i') in keys and keys[ord('i')] & p.KEY_IS_DOWN:
            self.left_speed = -10
            self.right_speed = -10

        if ord('k') in keys and keys[ord('k')] & p.KEY_IS_DOWN:
            self.left_speed = 10
            self.right_speed = 10

        # Turning is achieved by applying different speeds
        # to the left and right sides of the robot.
        if ord('j') in keys and keys[ord('j')] & p.KEY_IS_DOWN:
            self.left_speed = -6
            self.right_speed = 16

        if ord('l') in keys and keys[ord('l')] & p.KEY_IS_DOWN:
            self.left_speed = 6
            self.right_speed = -16

    def motor_noise(self, speed):

        # Add a random variation of ±2% to the motor speed.
        return np.random.uniform(0.98, 1.02) * speed

    def step_action(self):
        # 1. Update the LiDAR

        # LiDAR does not need to run every frame.
        # A new measurement is taken every 10 steps.
        if self.lidar_counter >= 10:

            distances, angles, object_ids, person_distance = self.ray_cast_lidar()

            # Reset the LiDAR counter
            self.lidar_counter = 0

            # If a person is detected within 1 m,
            # activate the emergency mode.
            if person_distance is not None and person_distance < 1:

                self.person_detected = True

                print("PERSON DETECTED! Distance:", round(person_distance, 2), "m")

            else:
                # If the person is outside the safety distance,
                # the robot can continue.
                self.person_detected = False

        # 2. Stop the robot if a person is too close
        if self.person_detected:

            # Stop both sides of the robot
            self.left_speed = 0
            self.right_speed = 0

        else:
            # 3. Pause between movements

            if self.pause_steps > 0:

                self.left_speed = 0
                self.right_speed = 0

                self.pause_steps -= 1

            else:
                # 4. Start the next route action

                if self.action is None and self.route_index < len(self.route):

                    self.start_next_action()

                # 5. Execute the current action
                if self.action_steps > 0:

                    if self.action == "forward":

                        self.left_speed = -10
                        self.right_speed = -10

                    elif self.action == "backward":

                        self.left_speed = 10
                        self.right_speed = 10

                    elif self.action == "left":

                        self.left_speed = -6
                        self.right_speed = 16

                    elif self.action == "right":

                        self.left_speed = 6
                        self.right_speed = -16

                    elif self.action == "wait":
                        
                        self.left_speed = 0
                        self.right_speed = 0

                    # Consume one action step
                    self.action_steps -= 1

                else:

                    # The action is complete.
                    self.left_speed = 0
                    self.right_speed = 0

                    self.action = None

                    # Short pause before the next movement
                    self.pause_steps = 60

        # 6. Apply motor noise

        # Noise represents small variations
        # in the actual motor speed.
        noisy_right = self.motor_noise(self.right_speed)
        noisy_left = self.motor_noise(self.left_speed)

        # 7. Wheel indices
        # Right wheels
        right_wheels = [0, 2]

        # Left wheels
        left_wheels = [1, 3]

        # 8. Apply speed to the right wheels
        for joint_id in right_wheels:

            p.setJointMotorControl2(
                bodyUniqueId=self.agv_id,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,

                # Apply the noisy speed
                # targetVelocity=noisy_right,
                targetVelocity=self.right_speed,

                force=50
            )

        # 9. Apply speed to the left wheels
        for joint_id in left_wheels:

            p.setJointMotorControl2(
                bodyUniqueId=self.agv_id,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,

                # Apply the noisy speed
                targetVelocity=noisy_left,
                # targetVelocity=self.left_speed,

                force=50
            )

        # 10. Update the LiDAR counter
        self.lidar_counter += 1