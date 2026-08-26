import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


class LargeNavigationMap:
    """Procedurally generated continuous navigation map backed by a grid SDF."""

    def __init__(self, width=100, height=100, seed=0, agent_radius=0.35):
        if width < 40 or height < 40:
            raise ValueError("width and height must be at least 40")
        self.width = int(width)
        self.height = int(height)
        self.agent_radius = float(agent_radius)
        self.rng = np.random.default_rng(seed)
        self.occupancy = np.zeros((self.height, self.width), dtype=bool)
        self.passage_centers = []
        self.passage_axes = []
        self._build_map()
        self._refresh_distance_field()

    def _open_gap(self, axis, fixed, centre, half_width=3):
        if axis == "vertical":
            low = max(1, int(centre) - half_width)
            high = min(self.height - 1, int(centre) + half_width + 1)
            self.occupancy[low:high, fixed] = False
            self.passage_centers.append((float(fixed), float(centre)))
            self.passage_axes.append(axis)
        else:
            low = max(1, int(centre) - half_width)
            high = min(self.width - 1, int(centre) + half_width + 1)
            self.occupancy[fixed, low:high] = False
            self.passage_centers.append((float(centre), float(fixed)))
            self.passage_axes.append(axis)

    def _build_map(self):
        """Build a 4x4 multi-route map with wide rooms and narrow passages."""
        occ = self.occupancy
        occ[0, :] = True
        occ[-1, :] = True
        occ[:, 0] = True
        occ[:, -1] = True

        verticals = [self.width // 4, self.width // 2, 3 * self.width // 4]
        horizontals = [self.height // 4, self.height // 2, 3 * self.height // 4]
        for x in verticals:
            occ[1:-1, x] = True
        for y in horizontals:
            occ[y, 1:-1] = True

        vertical_gap_fractions = (
            (0.14, 0.62, 0.87),
            (0.10, 0.38, 0.84),
            (0.16, 0.58, 0.90),
        )
        horizontal_gap_fractions = (
            (0.12, 0.43, 0.88),
            (0.17, 0.66, 0.91),
            (0.09, 0.47, 0.82),
        )
        for x, fractions in zip(verticals, vertical_gap_fractions):
            for fraction in fractions:
                self._open_gap("vertical", x, fraction * (self.height - 1))
        for y, fractions in zip(horizontals, horizontal_gap_fractions):
            for fraction in fractions:
                self._open_gap("horizontal", y, fraction * (self.width - 1))

        # A few interior blocks make open rooms nontrivial without closing them.
        rectangles = [
            (8, 32, 17, 38),
            (33, 8, 40, 17),
            (58, 31, 69, 37),
            (80, 57, 90, 64),
            (31, 80, 41, 88),
            (56, 57, 64, 68),
        ]
        scale_x = self.width / 100.0
        scale_y = self.height / 100.0
        for x0, y0, x1, y1 in rectangles:
            sx0 = int(round(x0 * scale_x))
            sx1 = int(round(x1 * scale_x))
            sy0 = int(round(y0 * scale_y))
            sy1 = int(round(y1 * scale_y))
            occ[sy0:sy1, sx0:sx1] = True

    def _refresh_distance_field(self):
        # At a free cell centre, subtracting half a cell approximates distance
        # to the wall surface rather than distance to the wall-cell centre.
        self.clearance_grid = distance_transform_edt(~self.occupancy) - 0.5

    def clearance(self, points):
        points = np.asarray(points, dtype=np.float64)
        single = points.ndim == 1
        if single:
            points = points[None, :]
        coordinates = np.vstack([points[:, 1], points[:, 0]])
        values = map_coordinates(
            self.clearance_grid,
            coordinates,
            order=1,
            mode="constant",
            cval=-1.0,
        )
        return float(values[0]) if single else values

    def is_free(self, point, margin=0.0):
        return self.clearance(point) >= self.agent_radius + float(margin)

    def line_is_free(self, start, end, margin=0.0, resolution=0.25):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        distance = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(distance / resolution)) + 1)
        points = np.linspace(start, end, count)
        required = self.agent_radius + float(margin)
        return bool(np.all(self.clearance(points) >= required))

    def sample_free_position(self, rng=None, margin=0.6):
        rng = self.rng if rng is None else rng
        valid = np.argwhere(
            self.clearance_grid >= self.agent_radius + float(margin)
        )
        if len(valid) == 0:
            raise RuntimeError("Map has no free position with the requested margin")
        y, x = valid[int(rng.integers(len(valid)))]
        jitter = rng.uniform(-0.35, 0.35, size=2)
        point = np.array([x, y], dtype=np.float64) + jitter
        return point if self.is_free(point, margin=0.1) else np.array([x, y], float)

    def random_task(self, rng=None, min_distance=None):
        rng = self.rng if rng is None else rng
        if min_distance is None:
            min_distance = 0.55 * min(self.width, self.height)
        for _ in range(10_000):
            start = self.sample_free_position(rng)
            goal = self.sample_free_position(rng)
            if np.linalg.norm(goal - start) >= min_distance:
                return start, goal
        raise RuntimeError("Could not sample a sufficiently separated task")

    def block_random_passage(self, rng=None):
        """Optionally close one generated passage and update the SDF."""
        rng = self.rng if rng is None else rng
        passage_index = int(rng.integers(len(self.passage_centers)))
        x, y = self.passage_centers[passage_index]
        axis = self.passage_axes[passage_index]
        ix, iy = int(round(x)), int(round(y))
        radius = 3
        if axis == "vertical":
            self.occupancy[max(1, iy - radius):min(self.height - 1, iy + radius + 1), ix] = True
        else:
            self.occupancy[iy, max(1, ix - radius):min(self.width - 1, ix + radius + 1)] = True
        self._refresh_distance_field()
        return np.array([x, y], dtype=np.float64)
