
import numpy as np
import pybullet as p


class ParticleFilter:
    """
    Particle Filter for 2D robot localization.

    Motion model:
        Uses odometry increments with motion-dependent noise.

    Measurement model:
        Compares real LiDAR ranges with expected ranges from ScanableMap
        using a robust per-ray likelihood.

    Resampling:
        Uses systematic resampling only when the effective particle
        count falls below a configurable threshold.
    """

    def __init__(
        self,
        num_particles,
        scan_map,
        gaussian_weights=True,
        motion_noise_distance=0.01,
        motion_noise_angle=0.01,
        lidar_sigma=0.30,
        random_measurement_probability=0.05,
        resample_threshold=0.5,
        resample_position_noise=0.01,
        resample_angle_noise=0.01,
        random_seed=None,
    ):
        # Particle Filter configuration.
        self.num_particles = int(num_particles)
        self.scan_map = scan_map
        self.gaussian_weights = gaussian_weights

        # Motion model noise coefficients.
        self.motion_noise_distance = motion_noise_distance
        self.motion_noise_angle = motion_noise_angle

        # LiDAR measurement model parameters.
        self.lidar_sigma = max(float(lidar_sigma), 1e-6)
        self.random_measurement_probability = np.clip(
            random_measurement_probability, 1e-6, 0.99
        )

        # Resampling configuration.
        self.resample_threshold = resample_threshold
        self.resample_position_noise = resample_position_noise
        self.resample_angle_noise = resample_angle_noise

        # Random generator allows reproducible experiments.
        self.rng = np.random.default_rng(random_seed)

        # Particles are stored as [x, y, theta].
        self.particles = np.empty((0, 3), dtype=float)

        # Keep the last normalized weights for diagnostics.
        self.weights = np.empty(0, dtype=float)

        # PyBullet debug line IDs.
        self.debug_ids = []

    @staticmethod
    def normalize_angle(theta):
        """Normalize an angle to [-pi, pi]."""
        return np.arctan2(np.sin(theta), np.cos(theta))

    # ========================================================
    # Initialization
    # ========================================================

    def initialize(
        self,
        x,
        y,
        theta,
        position_spread=0.20,
        angle_spread=0.10,
    ):

        self.particles = np.empty(
            (self.num_particles, 3),
            dtype=float,
        )

        self.particles[:, 0] = (
            x
            + self.rng.uniform(
                -position_spread,
                position_spread,
                self.num_particles,
            )
        )

        self.particles[:, 1] = (
            y
            + self.rng.uniform(
                -position_spread,
                position_spread,
                self.num_particles,
            )
        )

        self.particles[:, 2] = self.normalize_angle(
            theta
            + self.rng.uniform(
                -angle_spread,
                angle_spread,
                self.num_particles,
            )
        )

        # Start with a uniform particle distribution.
        self.weights = np.full(
            self.num_particles,
            1.0 / self.num_particles,
        )

    # -----------------------------------------------------
    # Motion prediction
    # -----------------------------------------------------

    def predict(self, forward_distance, delta_theta):
        """
        Propagate particles using odometry increments.

        Noise increases with the amount of motion. The midpoint
        heading is used to approximate curved motion more accurately.
        """

        if len(self.particles) == 0:
            return

        distance = float(forward_distance)
        angle = float(delta_theta)

        # Motion-dependent standard deviations.
        distance_std = (
            self.motion_noise_distance
            * abs(distance)
            + 0.001
            * (abs(angle) > 1e-6)
        )

        angle_std = (
            self.motion_noise_angle
            * abs(angle)
            + 0.001
            * (abs(distance) > 1e-6)
        )

        distance_noise = self.rng.normal(
            0.0,
            distance_std,
            self.num_particles,
        )

        angle_noise = self.rng.normal(
            0.0,
            angle_std,
            self.num_particles,
        )

        noisy_distance = distance + distance_noise
        noisy_angle = angle + angle_noise

        # Integrate using the midpoint heading.
        old_theta = self.particles[:, 2]
        midpoint_theta = old_theta + noisy_angle / 2.0

        self.particles[:, 0] += (
            noisy_distance * np.cos(midpoint_theta)
        )

        self.particles[:, 1] += (
            noisy_distance * np.sin(midpoint_theta)
        )

        self.particles[:, 2] = self.normalize_angle(
            old_theta + noisy_angle
        )

    # ----------------------------------------------------------
    # LiDAR measurement likelihood
    # ----------------------------------------------------------

    def calculate_weights(self, real_distances, real_angles):

        if len(self.particles) == 0:
            return np.empty(0, dtype=float)

        real_distances = np.asarray(
            real_distances,
            dtype=float,
        ).reshape(-1)

        # The current ScanableMap interface returns ranges and angles.
        # We compare corresponding ranges, so ray ordering and FOV
        # must be consistent between the real and simulated LiDAR.
        if len(real_distances) == 0:
            return self._uniform_weights()

        valid_real = np.isfinite(real_distances)

        if not np.any(valid_real):
            return self._uniform_weights()

        real_distances = real_distances[valid_real]

        # Infer the random component's range from the configured map.
        max_range = max(
            float(self.scan_map.max_distance),
            1e-6,
        )

        # Prevent zero or negative measured ranges from causing
        # numerical problems.
        real_distances = np.clip(
            real_distances,
            0.0,
            max_range,
        )

        log_likelihoods = np.full(
            len(self.particles),
            -np.inf,
            dtype=float,
        )

        sigma = self.lidar_sigma
        random_prob = self.random_measurement_probability

        # Uniform range density for the random measurement component.
        random_density = 1.0 / max_range

        for i, (x, y, theta) in enumerate(self.particles):

            # Generate the expected scan from the occupancy map.
            try:
                expected_distances, _ = self.scan_map.scan(
                    float(x),
                    float(y),
                    float(theta),
                )
            except (ValueError, IndexError, FloatingPointError):
                continue

            expected_distances = np.asarray(
                expected_distances,
                dtype=float,
            ).reshape(-1)

            # The simulated and real scans must have matching ray counts.
            if len(expected_distances) != len(valid_real):
                continue

            expected_distances = expected_distances[valid_real]

            valid_expected = np.isfinite(expected_distances)

            if not np.all(valid_expected):
                continue

            expected_distances = np.clip(
                expected_distances,
                0.0,
                max_range,
            )

            # Per-ray range residuals.
            residuals = real_distances - expected_distances

            # Gaussian hit density.
            gaussian_density = (
                np.exp(
                    -0.5 * (residuals / sigma) ** 2
                )
                / (np.sqrt(2.0 * np.pi) * sigma)
            )

            # Robust mixture: tolerate some unexpected readings.
            mixture_density = (
                (1.0 - random_prob) * gaussian_density
                + random_prob * random_density
            )

            # Average log likelihood avoids numerical underflow
            # and makes the scale less sensitive to ray count.
            log_likelihoods[i] = np.mean(
                np.log(np.maximum(mixture_density, 1e-300))
            )

        # If every particle produced an invalid scan, avoid NaNs.
        finite = np.isfinite(log_likelihoods)

        if not np.any(finite):
            return self._uniform_weights()

        # Subtract the largest log likelihood before exponentiation.
        # This does not change the normalized weight ratios.
        max_log_likelihood = np.max(
            log_likelihoods[finite]
        )

        weights = np.zeros(
            len(self.particles),
            dtype=float,
        )

        weights[finite] = np.exp(
            log_likelihoods[finite] - max_log_likelihood
        )

        total = np.sum(weights)

        if not np.isfinite(total) or total <= 0.0:
            return self._uniform_weights()

        weights /= total
        self.weights = weights.copy()

        return weights

    def _uniform_weights(self):
        """Return uniform weights and store them."""
        weights = np.full(
            len(self.particles),
            1.0 / max(len(self.particles), 1),
        )

        self.weights = weights.copy()
        return weights

    # -----------------------------------------------------------
    # Effective particle count
    # -----------------------------------------------------------

    @staticmethod
    def effective_sample_size(weights):
        """Calculate the effective number of particles."""
        weights = np.asarray(weights, dtype=float)

        if len(weights) == 0:
            return 0.0

        total = np.sum(weights)

        if (
            not np.isfinite(total)
            or total <= 0.0
            or np.any(weights < 0.0)
        ):
            return 0.0

        weights = weights / total
        return 1.0 / np.sum(weights ** 2)

    # -----------------------------------------------------------
    # Systematic resampling
    # =-----------------------------------------------------------

    def resample(self, weights):

        if len(self.particles) == 0:
            return

        weights = np.asarray(weights, dtype=float)

        if len(weights) != len(self.particles):
            return

        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.sum(weights) <= 0.0
        ):
            weights = self._uniform_weights()
        else:
            weights = weights / np.sum(weights)

        effective_n = self.effective_sample_size(weights)

        # Keep the existing particles if the population is healthy.
        if effective_n >= self.resample_threshold * len(self.particles):
            self.weights = weights.copy()
            return

        count = len(self.particles)

        # Systematic resampling uses one random offset and evenly
        # spaced sample positions.
        positions = (
            self.rng.random() + np.arange(count)
        ) / count

        cumulative_weights = np.cumsum(weights)
        cumulative_weights[-1] = 1.0

        indices = np.searchsorted(
            cumulative_weights,
            positions,
            side="left",
        )

        new_particles = self.particles[indices].copy()

        # Add small roughening noise to reduce particle impoverishment.
        new_particles[:, 0] += self.rng.normal(
            0.0,
            self.resample_position_noise,
            count,
        )

        new_particles[:, 1] += self.rng.normal(
            0.0,
            self.resample_position_noise,
            count,
        )

        new_particles[:, 2] += self.rng.normal(
            0.0,
            self.resample_angle_noise,
            count,
        )

        new_particles[:, 2] = self.normalize_angle(
            new_particles[:, 2]
        )

        self.particles = new_particles

        # After resampling, each particle has equal weight.
        self.weights = np.full(
            count,
            1.0 / count,
        )

    # ========================================================
    # Pose estimation
    # ========================================================

    def estimate_pose(self, weights):
        """Estimate x, y and heading using weighted particles."""

        if len(self.particles) == 0:
            raise RuntimeError(
                "Particle Filter has not been initialized."
            )

        weights = np.asarray(weights, dtype=float)

        if (
            len(weights) != len(self.particles)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.sum(weights) <= 0.0
        ):
            weights = self._uniform_weights()
        else:
            weights = weights / np.sum(weights)

        x = np.sum(weights * self.particles[:, 0])
        y = np.sum(weights * self.particles[:, 1])

        # Circular mean avoids problems near -pi / +pi.
        sin_theta = np.sum(
            weights * np.sin(self.particles[:, 2])
        )

        cos_theta = np.sum(
            weights * np.cos(self.particles[:, 2])
        )

        theta = np.arctan2(
            sin_theta,
            cos_theta,
        )

        return float(x), float(y), float(theta)

    # ------------------------------------------------------
    # PyBullet particle visualization
    # ------------------------------------------------------

    def draw_particles(self, weights):

        if len(self.particles) == 0:
            return

        weights = np.asarray(weights, dtype=float)

        if len(weights) != len(self.particles):
            return

        if not np.all(np.isfinite(weights)):
            weights = self._uniform_weights()
        elif np.sum(weights) <= 0.0:
            weights = self._uniform_weights()
        else:
            weights = weights / np.sum(weights)

        z = 0.15
        heading_length = 0.10

        # Create debug lines only once.
        while len(self.debug_ids) < len(self.particles):
            debug_id = p.addUserDebugLine(
                [0, 0, z],
                [0, 0, z],
                [0, 0, 1],
                lineWidth=2,
                lifeTime=0,
            )

            self.debug_ids.append(debug_id)

        min_weight = np.min(weights)
        max_weight = np.max(weights)

        for i, ((x, y, theta), weight) in enumerate(
            zip(self.particles, weights)
        ):
            if max_weight > min_weight:
                normalized = (
                    (weight - min_weight)
                    / (max_weight - min_weight)
                )
            else:
                normalized = 0.5

            color = [
                float(normalized),
                0.2,
                float(1.0 - normalized),
            ]

            start = [
                float(x),
                float(y),
                z,
            ]

            end = [
                float(x + heading_length * np.cos(theta)),
                float(y + heading_length * np.sin(theta)),
                z,
            ]

            p.addUserDebugLine(
                start,
                end,
                lineColorRGB=color,
                lineWidth=2,
                lifeTime=0,
                replaceItemUniqueId=self.debug_ids[i],
            )

        # Remove lines if the number of particles has decreased.
        while len(self.debug_ids) > len(self.particles):
            debug_id = self.debug_ids.pop()
            p.removeUserDebugItem(debug_id)
