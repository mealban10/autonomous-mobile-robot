import pybullet as p
import random
import math
import numpy as np

class DemoRobot2W:
    #---------------------------------------------------------------------------------------------------
    # __init__(x, y)	Constructs the cart in PyBullet and sets up the wheels, supports, and physics.
    #---------------------------------------------------------------------------------------------------

    def __init__(self, x=0, y=0, motion_noise=0.05, noise_enabled=False):       
        # Dimensions (metres)
        chassis_radius = 0.24   # Radius of the circular chassis
        H = 0.20                # Height of the chassis
        R = 0.12                # Radius of the drive wheels
        T = 0.08                # Thickness of the wheels
        B = 0.06                # Radius of the support spheres
        support_x = 0.20        # Longitudinal distance from the spheres to the center

        # Chassis centre height: drive wheel bottoms at z = 0
        base_z = R + H / 2

        # Circular chassis
        chassis_col = p.createCollisionShape(p.GEOM_CYLINDER,radius=chassis_radius,height=H)
        chassis_vis = p.createVisualShape(p.GEOM_CYLINDER,radius=chassis_radius,length=H,rgbaColor=[0.2, 0.7, 0.4, 1.0])

        # Drive wheels
        wheel_col = p.createCollisionShape(p.GEOM_CYLINDER,radius=R,height=T)
        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER,radius=R,length=T,rgbaColor=[0.25, 0.25, 0.25, 1.0])

        # Fixed spherical supports
        support_col = p.createCollisionShape(p.GEOM_SPHERE,radius=B)
        support_vis = p.createVisualShape(p.GEOM_SPHERE,radius=B,rgbaColor=[0.85, 0.55, 0.15, 1.0])

        # Symmetric placement of the drive wheels
        wheel_y = chassis_radius + T / 2

        # Support sphere centres are B metres above the ground
        support_z = B - base_z
        wheel_orientation = p.getQuaternionFromEuler([1.57079632679, 0, 0])

        # Create robot
        self.agv_id = p.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=chassis_col,
            baseVisualShapeIndex=chassis_vis,
            basePosition=[x, y, base_z],

            linkMasses=[1.5, 1.5, 0.05, 0.05],
            linkCollisionShapeIndices=[
                wheel_col,
                wheel_col,
                support_col,
                support_col
            ],
            linkVisualShapeIndices=[
                wheel_vis,
                wheel_vis,
                support_vis,
                support_vis
            ],
            linkPositions=[
                [0, wheel_y, -H / 2],       # Right wheel
                [0, -wheel_y, -H / 2],      # Left wheel
                [support_x, 0, support_z],  # Front support
                [-support_x, 0, support_z]  # Rear support
            ],
            linkOrientations=[
                wheel_orientation,
                wheel_orientation,
                [0, 0, 0, 1],
                [0, 0, 0, 1]
            ],
            linkInertialFramePositions=[[0, 0, 0]] * 4,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
            linkParentIndices=[0, 0, 0, 0],
            linkJointTypes=[
                p.JOINT_REVOLUTE,
                p.JOINT_REVOLUTE,
                p.JOINT_FIXED,
                p.JOINT_FIXED
            ],
            linkJointAxis=[
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]
            ]
        )

        # Joint indices
        self.right_joint = 0
        self.left_joint = 1

        # Smooth-control state
        self.current_left = 0.0
        self.current_right = 0.0
        self.max_speed_change = 0.15

        # Chassis physics
        p.changeDynamics(self.agv_id, -1, lateralFriction=0.8, linearDamping=0.1, angularDamping=0.2)

        # Drive wheel physics
        for joint in [self.right_joint, self.left_joint]:
            p.changeDynamics(self.agv_id, joint, lateralFriction=1.5, rollingFriction=0.001, spinningFriction=0.001 )

            p.setJointMotorControl2(self.agv_id, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=50)

        # Low-friction fixed supports
        for joint in [2, 3]:
            p.changeDynamics(self.agv_id, joint, lateralFriction=0.02, rollingFriction=0.001, spinningFriction=0.001)

        # Motion noise configuration
        self.motion_noise = max(0.0, float(motion_noise))
        self.noise_enabled = bool(noise_enabled)

        print("Robot ID:", self.agv_id)
        print("Number of joints:", p.getNumJoints(self.agv_id))

    #----------------------------------------------------------------------------------------------
    # set_wheel_speeds(left, right)	Sends target speeds to the left and right wheels. Can apply noise if enabled.
    #---------------------------------------------------------------------------------------------    
    
    def set_wheel_speeds(self, left, right):        
        # Apply motion noise only when enabled
        if self.noise_enabled:
            left *= random.uniform(1.0 - self.motion_noise, 1.0 + self.motion_noise)
            right *= random.uniform(1.0 - self.motion_noise, 1.0 + self.motion_noise)

        # Apply wheel velocity commands
        p.setJointMotorControl2(self.agv_id, self.left_joint, p.VELOCITY_CONTROL, targetVelocity=left, force=50,)
        p.setJointMotorControl2(self.agv_id, self.right_joint, p.VELOCITY_CONTROL, targetVelocity=right, force=50,)

    #----------------------------------------------------------------------------------------------
    # ray_cast_lidar(...)	Simulates a LiDAR and returns distances to obstacles as well as the angles of the rays.
    #---------------------------------------------------------------------------------------------    

    def ray_cast_lidar(self, num_rays=21, fov=2.0, max_distance=10.0, z_offset=0.05, ):
        # Get the robot's current position and orientation.
        position, orientation = p.getBasePositionAndOrientation(self.agv_id)
        _, _, yaw = p.getEulerFromQuaternion(orientation)

        # Generate ray angles across the field of view.
        relative_angles = [-fov / 2 + i * fov / num_rays for i in range(num_rays) ]
        angles = [yaw + angle for angle in relative_angles]

        # Set the ray origin slightly above the ground.
        ray_start = [ position[0], position[1], position[2] + z_offset, ]

        ray_starts = []
        ray_ends = []

        for angle in angles:
            ray_starts.append(ray_start)
            ray_ends.append([ position[0] + max_distance * math.cos(angle), position[1] + max_distance * math.sin(angle), ray_start[2], ])

        # Cast all rays in a single batch.
        results = p.rayTestBatch( ray_starts, ray_ends, numThreads=0, )

        distances = []

        for result in results:
            hit_object = result[0]
            hit_fraction = result[2]

            if hit_object == -1:
                distance = max_distance
            elif hit_object == self.agv_id:
                # Ignore the robot itself if a ray hits it.
                distance = max_distance
            else:
                distance = hit_fraction * max_distance

            distances.append(distance)

        return np.array(distances), np.array(angles)

    #----------------------------------------------------------------------------------------------
    # stop()	Commands the wheels to stop and resets the speed variables used for manual control.
    #---------------------------------------------------------------------------------------------    

    def stop(self):
        self.current_left = 0.0
        self.current_right = 0.0
        self.set_wheel_speeds(0, 0)

    #----------------------------------------------------------------------------------------------
    # user_control(keys)	Allows the cart to be driven using the keyboard, with progressive acceleration and braking.
    #---------------------------------------------------------------------------------------------    
  
    def user_control(self, keys):
        target_left = 0.0
        target_right = 0.0

        # I: forward
        if ord('i') in keys and keys[ord('i')] & p.KEY_IS_DOWN:
            target_left = -3.5
            target_right = -3.5

        # K: backward
        elif ord('k') in keys and keys[ord('k')] & p.KEY_IS_DOWN:
            target_left = 3.5
            target_right = 3.5

        # J: turn left
        elif ord('j') in keys and keys[ord('j')] & p.KEY_IS_DOWN:
            target_left = -1.5
            target_right = 1.5

        # L: turn right
        elif ord('l') in keys and keys[ord('l')] & p.KEY_IS_DOWN:
            target_left = 1.5
            target_right = -1.5

        # Smooth acceleration and braking
        change = self.max_speed_change

        self.current_left += max( -change, min(change, target_left - self.current_left))
        self.current_right += max( -change, min(change, target_right - self.current_right))
        self.set_wheel_speeds( self.current_left, self.current_right)



