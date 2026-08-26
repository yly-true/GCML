import numpy as np
from scipy.interpolate import splprep, splev


class SplineRolloutPlanner:
    """Sample and score smooth B-spline curves through GCML landmarks."""

    def __init__(
        self,
        navigation_map,
        candidates=96,
        control_spacing=4.0,
        sample_spacing=0.20,
        clearance_margin=0.18,
        seed=0,
    ):
        self.map = navigation_map
        self.candidates = int(candidates)
        self.control_spacing = float(control_spacing)
        self.sample_spacing = float(sample_spacing)
        self.clearance_margin = float(clearance_margin)
        self.rng = np.random.default_rng(seed)

    def _densify(self, waypoints):
        dense = [np.asarray(waypoints[0], dtype=np.float64)]
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            distance = float(np.linalg.norm(end - start))
            count = max(1, int(np.ceil(distance / self.control_spacing)))
            for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
                dense.append((1.0 - fraction) * start + fraction * end)
        return np.asarray(dense)

    def _curve_from_controls(self, controls, smoothing):
        if len(controls) <= 2:
            distance = np.linalg.norm(controls[-1] - controls[0])
            count = max(2, int(np.ceil(distance / self.sample_spacing)) + 1)
            return np.linspace(controls[0], controls[-1], count)

        segment_length = np.linalg.norm(np.diff(controls, axis=0), axis=1)
        approximate_length = float(np.sum(segment_length))
        sample_count = max(
            80, int(np.ceil(approximate_length / self.sample_spacing)) + 1
        )
        degree = min(3, len(controls) - 1)
        spline, _ = splprep(
            controls.T,
            s=float(smoothing) * len(controls),
            k=degree,
        )
        parameter = np.linspace(0.0, 1.0, sample_count)
        curve = np.column_stack(splev(parameter, spline))
        curve[0] = controls[0]
        curve[-1] = controls[-1]
        return curve

    def _score(self, curve):
        clearance = self.map.clearance(curve)
        required = self.map.agent_radius + self.clearance_margin
        violation = np.maximum(required - clearance, 0.0)
        collision_cost = float(np.sum(violation * violation))

        segments = np.diff(curve, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        path_length = float(np.sum(lengths))
        directions = segments / np.maximum(lengths[:, None], 1e-9)
        curvature = float(
            np.sum(np.linalg.norm(np.diff(directions, axis=0), axis=1) ** 2)
        )
        free_clearance = np.maximum(clearance - self.map.agent_radius, 0.0)
        proximity_cost = float(np.mean(np.exp(-free_clearance / 0.8)))
        score = (
            100_000.0 * collision_cost
            + path_length
            + 0.35 * curvature
            + 2.0 * proximity_cost
        )
        return {
            "score": score,
            "safe": bool(np.all(clearance >= self.map.agent_radius)),
            "path_length": path_length,
            "curvature": curvature,
            "minimum_clearance": float(np.min(clearance)),
        }

    def plan(self, waypoints):
        waypoints = np.asarray(waypoints, dtype=np.float64)
        controls = self._densify(waypoints)
        best_curve = None
        best_metrics = None

        for candidate_index in range(self.candidates):
            candidate = controls.copy()
            if candidate_index == 0:
                jitter_scale = 0.0
                smoothing = 0.0
            else:
                jitter_scale = self.rng.uniform(0.0, 0.9)
                smoothing = self.rng.uniform(0.0, 0.30)

            if len(candidate) > 2 and jitter_scale > 0.0:
                clearance = self.map.clearance(candidate)
                movable = clearance > self.map.agent_radius + 2.0
                movable[[0, -1]] = False
                perturbation = self.rng.normal(
                    0.0, jitter_scale, size=candidate.shape
                )
                proposed = candidate + perturbation * movable[:, None]
                for index in np.flatnonzero(movable):
                    if self.map.is_free(proposed[index], margin=0.25):
                        candidate[index] = proposed[index]

            try:
                curve = self._curve_from_controls(candidate, smoothing)
            except ValueError:
                continue
            metrics = self._score(curve)
            key = (not metrics["safe"], metrics["score"])
            if best_metrics is None:
                best_curve, best_metrics, best_key = curve, metrics, key
            elif key < best_key:
                best_curve, best_metrics, best_key = curve, metrics, key

        if best_curve is None:
            raise RuntimeError("B-spline candidate generation failed")
        best_metrics["candidate_count"] = self.candidates
        return best_curve, best_metrics
