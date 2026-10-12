import numpy as np
import pybullet as p


class ParticleFilter:
    def __init__(self, num_particles, scan_map, gaussian_weights=False):
        # Particle filter configuration
        self.num_particles = num_particles
        self.scan_map = scan_map
        self.gaussian_weights = gaussian_weights

        self.particles = []
        self.debug_ids = []

    def initialize(self, x, y, theta):
        # Initialize particles around the initial pose
        self.particles = []

        for i in range(self.num_particles):
            px = x + np.random.uniform(-0.2, 0.2)
            py = y + np.random.uniform(-0.2, 0.2)
            ptheta = theta + np.random.uniform(-0.1, 0.1)

            self.particles.append([px, py, ptheta])

    def predict(self, forward_distance, delta_theta):
        # Predict particle motion using odometry

        for particle in self.particles:

            x, y, theta = particle

            # Add motion noise only when the robot moves
            if abs(forward_distance) < 1e-6 and abs(delta_theta) < 1e-6:
                distance_noise = 0
                angle_noise = 0
            else:
                distance_noise = np.random.normal(0, 0.01)
                angle_noise = np.random.normal(0, 0.01)

            distance = forward_distance + distance_noise
            angle = delta_theta + angle_noise

            x += distance * np.cos(theta)
            y += distance * np.sin(theta)
            theta += angle

            # Normalize angle to [-pi, pi]
            theta = np.arctan2(np.sin(theta), np.cos(theta))

            particle[0] = x
            particle[1] = y
            particle[2] = theta

    def draw_particles(self, weights):
        # Remove previous particle visualization
        for debug_id in self.debug_ids:
            p.removeUserDebugItem(debug_id)

        self.debug_ids = []

        min_weight = np.min(weights)
        max_weight = np.max(weights)

        for (x, y, theta), weight in zip(self.particles, weights):

            # Normalize particle weights for visualization
            if max_weight > min_weight:
                normalized = (weight - min_weight) / (max_weight - min_weight)
            else:
                normalized = 0.5

            red = normalized
            blue = 1.0 - normalized
            green = 0.2

            # Draw particle position and heading
            debug_id = p.addUserDebugLine(
                [x, y, 0.3],
                [x + 0.05 * np.cos(theta),
                 y + 0.05 * np.sin(theta),
                 0.3],
                lineColorRGB=[red, green, blue],
                lineWidth=5
            )

            self.debug_ids.append(debug_id)

    def calculate_weights(self, real_distances, real_angles):
        # Calculate particle weights from LiDAR measurements
        weights = []

        for x, y, theta in self.particles:

            # Generate expected LiDAR scan from the map
            expected_distances, _ = self.scan_map.scan(x, y, theta)

            # Compare measured and expected distances
            error = np.mean(np.abs(real_distances - expected_distances))

            # Convert scan error into a particle weight
            if self.gaussian_weights:
                sigma = 0.8
                weight = np.exp(-(error**2) / (2 * sigma**2))
            else:
                weight = 1.0 / (error + 0.001)

            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        total = np.sum(weights)

        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(self.num_particles) / self.num_particles

        return weights

    def resample(self, weights):
        # Resample particles according to their weights
        new_particles = []
        selected = []

        for i in range(self.num_particles):

            index = np.random.choice(len(self.particles), p=weights)

            selected.append(index)

            x, y, theta = self.particles[index]

            # Add small noise after resampling
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            theta += np.random.normal(0, 0.01)

            new_particles.append([x, y, theta])

        # print("Selected:", selected)

        self.particles = new_particles

    def estimate_pose(self, weights):
        # Estimate pose using the weighted particle distribution
        weights = np.array(weights)

        x = sum(
            weights[i] * self.particles[i][0]
            for i in range(self.num_particles)
        )

        y = sum(
            weights[i] * self.particles[i][1]
            for i in range(self.num_particles)
        )

        # Estimate orientation using weighted sine and cosine
        sin_theta = sum(
            weights[i] * np.sin(self.particles[i][2])
            for i in range(self.num_particles)
        )

        cos_theta = sum(
            weights[i] * np.cos(self.particles[i][2])
            for i in range(self.num_particles)
        )

        theta = np.arctan2(sin_theta, cos_theta)

        return x, y, theta