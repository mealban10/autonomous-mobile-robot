import math
import numpy as np


class DynamicObstacleTracker:

    #---------------------------------------------------------------------------------------------------
    # __init__(...)    Initializes obstacle detection, tracking and safety parameters.
    #---------------------------------------------------------------------------------------------------
    def __init__(
        self,
        distance_threshold=0.30,
        max_detection_distance=3.5,
        association_distance=0.8,
        max_track_age=1.0,
        stop_distance=0.9,
        slow_distance=1.8,
        front_angle=1.0,
        min_group_rays=2,
        stop_release_distance=1.2,
    ):
        self.distance_threshold = distance_threshold
        self.max_detection_distance = max_detection_distance
        self.association_distance = association_distance
        self.max_track_age = max_track_age

        self.stop_distance = stop_distance
        self.slow_distance = slow_distance
        self.stop_release_distance = max(stop_release_distance, stop_distance)

        self.front_angle = front_angle
        self.min_group_rays = max(1, int(min_group_rays))

        self.tracks = []
        self.next_track_id = 1
        self.last_time = None
        self.stop_latched = False

    #---------------------------------------------------------------------------------------------------
    # _angle_difference(...)    Calculates the shortest angular difference between two angles.
    #---------------------------------------------------------------------------------------------------
    @staticmethod
    def _angle_difference(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    #---------------------------------------------------------------------------------------------------
    # _extract_detections(...)    Identifies potential obstacles by comparing real and expected LiDAR distances.
    #---------------------------------------------------------------------------------------------------
    def _extract_detections(
        self,
        real_distances,
        real_angles,
        expected_distances,
        robot_pose,
    ):
        x, y, _ = robot_pose

        real_distances = np.asarray(real_distances, dtype=float)
        real_angles = np.asarray(real_angles, dtype=float)
        expected_distances = np.asarray(expected_distances, dtype=float)

        if not (
            len(real_distances) == len(real_angles) == len(expected_distances)
        ):
            raise ValueError(
                "real_distances, real_angles and expected_distances "
                "must have the same length"
            )

        difference = expected_distances - real_distances
        valid = (
            (difference > self.distance_threshold)
            & (real_distances <= self.max_detection_distance)
            & (real_distances > 0.05)
            & np.isfinite(real_distances)
            & np.isfinite(real_angles)
            & np.isfinite(expected_distances)
        )

        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            return []

        # Merge consecutive rays so one object becomes one detection.
        groups = []
        current = [int(indices[0])]
        for raw_idx in indices[1:]:
            idx = int(raw_idx)
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
        groups.append(current)

        detections = []
        for group in groups:
            if len(group) < self.min_group_rays:
                continue

            # Convert each ray return to world Cartesian coordinates.
            points = [
                (
                    x + real_distances[i] * math.cos(real_angles[i]),
                    y + real_distances[i] * math.sin(real_angles[i]),
                )
                for i in group
            ]

            px = float(np.mean([point[0] for point in points]))
            py = float(np.mean([point[1] for point in points]))
            distance = float(np.mean(real_distances[group]))
            bearing = math.atan2(py - y, px - x)

            detections.append(
                {
                    "x": px,
                    "y": py,
                    "distance": distance,
                    "bearing": bearing,
                    "ray_indices": group,
                }
            )

        return detections

    #---------------------------------------------------------------------------------------------------
    # update(...)    Updates obstacle tracks, estimates object motion and determines the robot's safety state.
    #---------------------------------------------------------------------------------------------------
    def update(
        self,
        real_distances,
        real_angles,
        expected_distances,
        robot_pose,
        sim_time,
    ):

        x, y, theta = robot_pose

        detections = self._extract_detections(
            real_distances,
            real_angles,
            expected_distances,
            robot_pose,
        )

        self.last_time = sim_time

        for track in self.tracks:
            track["matched"] = False

        # Associate each detection with at most one existing track.
        for detection in detections:
            candidates = [
                track
                for track in self.tracks
                if not track["matched"]
                and math.hypot(
                    detection["x"] - track["x"],
                    detection["y"] - track["y"],
                ) <= self.association_distance
            ]

            if candidates:
                track = min(
                    candidates,
                    key=lambda item: math.hypot(
                        detection["x"] - item["x"],
                        detection["y"] - item["y"],
                    ),
                )

                elapsed = sim_time - track["last_seen"]
                if elapsed > 1e-6:
                    measured_vx = (detection["x"] - track["x"]) / elapsed
                    measured_vy = (detection["y"] - track["y"]) / elapsed
                    # Smooth estimated object velocity.
                    track["vx"] = 0.5 * track["vx"] + 0.5 * measured_vx
                    track["vy"] = 0.5 * track["vy"] + 0.5 * measured_vy

                track["x"] = detection["x"]
                track["y"] = detection["y"]
                track["distance"] = detection["distance"]
                track["bearing"] = detection["bearing"]
                track["last_seen"] = sim_time
                track["hits"] += 1
                track["consecutive_hits"] += 1
                track["matched"] = True
                track["ray_indices"] = detection["ray_indices"]

            else:
                self.tracks.append(
                    {
                        "id": self.next_track_id,
                        "x": detection["x"],
                        "y": detection["y"],
                        "vx": 0.0,
                        "vy": 0.0,
                        "distance": detection["distance"],
                        "bearing": detection["bearing"],
                        "last_seen": sim_time,
                        "hits": 1,
                        "consecutive_hits": 1,
                        "matched": True,
                        "ray_indices": detection["ray_indices"],
                    }
                )
                self.next_track_id += 1

        # Reset consecutive detections when a track is not observed.
        for track in self.tracks:
            if not track["matched"]:
                track["consecutive_hits"] = 0

        # Remove tracks that have not been observed recently.
        self.tracks = [
            track
            for track in self.tracks
            if sim_time - track["last_seen"] <= self.max_track_age
        ]

        # Find the nearest confirmed object inside the front corridor.
        nearest_front_distance = float("inf")
        nearest_front_track = None

        for track in self.tracks:
            dx = track["x"] - x
            dy = track["y"] - y
            distance = math.hypot(dx, dy)

            relative_bearing = self._angle_difference(
                math.atan2(dy, dx), theta
            )

            if abs(relative_bearing) > self.front_angle:
                continue

            if track["consecutive_hits"] < 2:
                continue

            if distance < nearest_front_distance:
                nearest_front_distance = distance
                nearest_front_track = track

        # Check confirmed tracks within the robot's front corridor.
        for track in self.tracks:
            dx = track["x"] - x
            dy = track["y"] - y
            distance = math.hypot(dx, dy)

            relative_bearing = self._angle_difference(
                math.atan2(dy, dx), theta
            )

            if abs(relative_bearing) > self.front_angle:
                continue

            if track["consecutive_hits"] < 2:
                continue

            nearest_front_distance = min(
                nearest_front_distance,
                distance,
            )

        # Stop hysteresis: require additional clearance before resuming movement.
        if self.stop_latched:
            if nearest_front_distance > self.stop_release_distance:
                self.stop_latched = False
        elif nearest_front_distance <= self.stop_distance:
            self.stop_latched = True

        # Determine the safety state and corresponding speed factor.
        if self.stop_latched:
            safety_state = "STOP"
            should_stop = True
            speed_scale = 0.0
        elif nearest_front_distance <= self.slow_distance:
            safety_state = "SLOW"
            should_stop = False
            speed_scale = 0.4
        else:
            safety_state = "CLEAR"
            should_stop = False
            speed_scale = 1.0

        # Assign visual states to rays belonging to the nearest confirmed obstacle.
        ray_states = {}

        if nearest_front_track is not None:
            if self.stop_latched:
                ray_state = "STOP"
            elif nearest_front_distance <= self.slow_distance:
                ray_state = "SLOW"
            else:
                ray_state = None

            if ray_state is not None:
                for index in nearest_front_track["ray_indices"]:
                    ray_states[index] = ray_state

        # Return updated tracks and safety information.
        return list(self.tracks), {
            "stop": should_stop,
            "speed_scale": speed_scale,
            "detections": detections,
            "state": safety_state,
            "nearest_distance": nearest_front_distance,
            "ray_states": ray_states,
        }