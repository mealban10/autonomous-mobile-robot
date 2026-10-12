import time
import csv
import pybullet as p
import numpy as np
import pygame
import matplotlib.pyplot as plt

import utils.world as W
import utils.demo_robot as R
import utils.demo_person as P

from utils.demo_odometry import DifferentialDriveOdometry, debug_text
from utils.ScannableMap import ScanableMap
from utils.particle_filter import ParticleFilter

# Define the parameters used for the particle filter experiments
NUM_PARTICLES = 40
PF_UPDATE_INTERVAL = 0.5  # seconds
CSV_FILE = "lab4_particles_test01.csv"

# Create the simulation world and load the maze
wld = W.BaseWorld()
wld.map_from_file("maps/demoMap")
wld.texture_walls()

# Create the robot at the initial position
r = R.DemoRobot(x=1.5, y=0.5)

# Pause briefly before starting the simulation
#time.sleep(1)

# Initialize wheel-based odometry
odom = DifferentialDriveOdometry(
    wheel_radius=0.12,
    wheel_base=0.57,
    left_wheel_joints=[1, 3],
    right_wheel_joints=[0, 2]
)

odom.reset(1.5, 0.5, 0)  #initial pose of the robot for the odometry


# Configure the map scanner used by the particle filter
smap = ScanableMap(
    map_dimensions=(25, 19),
    fov=np.degrees(2.0),
    max_distance=10,
    num_rays=21,
    scale=1
)

smap.load_map("maps/demoMap")


# Initialize the particle filter with the selected number of particles
pf = ParticleFilter(
    num_particles=NUM_PARTICLES,
    scan_map=smap,
    gaussian_weights=True
)

pf.initialize(1.5, 0.5, 0) #initial pose of the robot for the particle filter

# Store the estimated, odometry and ground truth trajectories
pf_history = []
odom_history = []
true_history = []
odom_error_history = []
pf_error_history = []

# Create the CSV file and define the data columns
csv_file = open(CSV_FILE, "w", newline="")

csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time","true_x","true_y","odom_x","odom_y","pf_x","pf_y","true_theta","odom_theta","pf_theta"])

# Compare the real LiDAR scan with the map-based scan
def compare_scans():
    pos, orn = p.getBasePositionAndOrientation(r.agv_id)
    theta = p.getEulerFromQuaternion(orn)[2]
    real_distances, real_angles, _, _ = r.ray_cast_lidar()
    map_distances, map_angles = smap.scan(pos[0],pos[1],theta)

compare_scans()

# Create the moving people used as obstacles

person1 = P.DemoPerson()
person1.set_position(6, 3.5)
person2 = P.DemoPerson()
person2.set_position(4, 8.5)
person3 = P.DemoPerson()
person3.set_position(10, 11)

r.set_person_ids([
    person1.objectId,
    person2.objectId,
    person3.objectId
])

# Track the elapsed time between particle filter updates
pf_timer = 0.0
freq_hz = 240.0

# Update odometry continuously and run the particle filter at the selected interval
def update_odometry():

    global pf_timer

    # Update the robot odometry from the wheel movements
    motion = odom.update(r.agv_id)

    if motion is not None:

        forward_distance, delta_theta = motion

        if (abs(forward_distance) > 1e-6 or abs(delta_theta) > 1e-6):
            pf.predict(forward_distance,delta_theta)

    # Accumulate simulation time at 240 Hz
    pf_timer += 1 / freq_hz

    # Run the localization update when the selected interval is reached
    if pf_timer >= PF_UPDATE_INTERVAL:

        # Get the robot ground truth position and orientation
        pos, orn = p.getBasePositionAndOrientation(r.agv_id)
        theta = p.getEulerFromQuaternion(orn)[2]

        # Obtain the current LiDAR measurement from the robot
        real_distances, real_angles, _, _ = (r.ray_cast_lidar())

        # Calculate particle weights from the LiDAR measurement
        weights = pf.calculate_weights(real_distances,real_angles)

        # Estimate the robot pose from the weighted particles
        estimate_x, estimate_y, estimate_theta = (pf.estimate_pose(weights))

        # Calculate the position error of odometry and particle filter
        odom_error = np.sqrt((odom.x - pos[0]) ** 2 + (odom.y - pos[1]) ** 2)
        pf_error = np.sqrt((estimate_x - pos[0]) ** 2 + (estimate_y - pos[1]) ** 2)

        # Store the localization results for later plotting
        pf_history.append([estimate_x, estimate_y])

        odom_history.append([odom.x, odom.y])
        true_history.append([pos[0],pos[1]])

        odom_error_history.append(odom_error)
        pf_error_history.append(pf_error)

        # Save the current localization results to the CSV file
        csv_writer.writerow([len(pf_history) * PF_UPDATE_INTERVAL,pos[0],pos[1],odom.x,odom.y,
        estimate_x,estimate_y,theta,odom.theta,estimate_theta])

        csv_file.flush()

        # Find the particle with the highest weight
        best_index = np.argmax(weights)

        best_particle = pf.particles[best_index]

        # Display the particle filter estimate
        print("PF promedio:", np.round([estimate_x, estimate_y, np.degrees(estimate_theta)], 2))

        # Display the best individual particle
        print("PF mejor:",np.round([best_particle[0],best_particle[1],np.degrees(best_particle[2])],2))

        # Display the ground truth robot pose
        print("True:",np.round([pos[0],pos[1],np.degrees(theta)],2))

        # Display the particle weights
        print("Pesos:",np.round(weights, 3))

        # Display the sum of the particle weights
        print("Suma:",np.sum(weights))

        # Draw the particles and resample them for the next iteration
        pf.draw_particles(weights)
        pf.resample(weights)

        # Reset the particle filter timer
        pf_timer = 0.0

        # Show the trajectory after enough measurements
        if len(pf_history) >= 90:
            plot_trajectory()


# Plot the ground truth, odometry and particle filter trajectories
def plot_trajectory():

    pf_data = np.array(pf_history)
    odom_data = np.array(odom_history)
    true = np.array(true_history)

    plt.figure(figsize=(8, 6))

    plt.plot(true[:, 0],true[:, 1],label="Ground Truth")
    plt.plot(odom_data[:, 0],odom_data[:, 1],label="Odometry")
    plt.plot(pf_data[:, 0],pf_data[:, 1],label="Particle Filter")

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Robot trajectory")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()


# Define the calibrated robot route through the maze
r.set_route([
    ("left", 85),
    ("forward", 0.95),

    ("right", 78),
    ("forward", 1.7),

    ("left", 90),
    ("forward", 1.8),

    ("right", 70),
    ("forward", 3.6),

    ("left", 80),
    ("forward", 3.7),

    ("left", 85),
    ("forward", 1.7),

    ("right", 85),
    ("forward", 2.0),

    ("right", 70),
    ("forward", 5.5)
])

# Define the movement paths followed by each person
g1 = P.ValidPathGraph()
g1.add_positions([(6, 3.6),(6, 2)])
g1.add_connection((6, 3.6),(6, 2))

g2 = P.ValidPathGraph()
g2.add_positions([(4, 8.5),(5.5, 8.5)])
g2.add_connection((4, 8.5),(5.5, 8.5))

g3 = P.ValidPathGraph()
g3.add_positions([(10, 11),(10, 9)])
g3.add_connection((10, 11),(10, 9))

person1.set_movement_graph(g1)
person2.set_movement_graph(g2)
person3.set_movement_graph(g3)

# Register the robot, localization and person callbacks
wld.add_step_callback(r.step_action)
wld.add_keyboard_callback(r.user_control)

wld.add_step_callback(update_odometry)

wld.add_step_callback(person1.step_action)
wld.add_step_callback(person2.step_action)
wld.add_step_callback(person3.step_action)


# Run the simulation and close the CSV file when it is stopped
try:

    while True:
        wld.simStep()

except KeyboardInterrupt:

    print("\nSimulation stopped.")

finally:

    csv_file.close()
    wld.end()