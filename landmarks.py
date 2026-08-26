import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


class LandmarkGraph:
    """Sparse visibility graph whose size follows map structure, not cell count."""

    def __init__(
        self,
        navigation_map,
        coverage_radius=8.0,
        connection_radius=19.0,
        max_landmarks=100,
        max_neighbors=8,
        seed=0,
    ):
        self.map = navigation_map
        self.rng = np.random.default_rng(seed)
        self.positions = self._select_landmarks(
            coverage_radius=coverage_radius,
            max_landmarks=max_landmarks,
        )
        self.adjacency = self._build_adjacency(
            connection_radius=connection_radius,
            max_neighbors=max_neighbors,
        )
        self._build_directed_actions()
        weighted = np.where(self.adjacency, self._pairwise_distances(), 0.0)
        self.shortest_distances = shortest_path(
            csr_matrix(weighted), directed=False, unweighted=False
        )

    @property
    def num_nodes(self):
        return len(self.positions)

    @property
    def num_actions(self):
        return len(self.directed_edges)

    def _select_landmarks(self, coverage_radius, max_landmarks):
        mandatory = []
        for point in self.map.passage_centers:
            point = np.asarray(point, dtype=np.float64)
            if self.map.is_free(point, margin=0.05):
                if not any(np.linalg.norm(point - old) < 1.0 for old in mandatory):
                    mandatory.append(point)

        stride = max(2, int(round(coverage_radius / 3.0)))
        candidates = []
        for y in range(2, self.map.height - 2, stride):
            for x in range(2, self.map.width - 2, stride):
                point = np.array([x, y], dtype=np.float64)
                if self.map.is_free(point, margin=0.7):
                    candidates.append(point)
        if not candidates:
            raise RuntimeError("No free landmark candidates")
        candidates = np.asarray(candidates)

        selected = list(mandatory)
        if not selected:
            selected.append(candidates[np.argmax(self.map.clearance(candidates))])

        while len(selected) < max_landmarks:
            current = np.asarray(selected)
            distance = np.linalg.norm(
                candidates[:, None, :] - current[None, :, :], axis=2
            )
            nearest = np.min(distance, axis=1)
            index = int(np.argmax(nearest))
            if nearest[index] <= coverage_radius:
                break
            selected.append(candidates[index])

        return np.asarray(selected, dtype=np.float64)

    def _pairwise_distances(self):
        delta = self.positions[:, None, :] - self.positions[None, :, :]
        return np.linalg.norm(delta, axis=2)

    def _build_adjacency(self, connection_radius, max_neighbors):
        count = len(self.positions)
        distances = self._pairwise_distances()

        for radius_scale in (1.0, 1.3, 1.7, 2.2, 3.0):
            adjacency = np.zeros((count, count), dtype=bool)
            radius = connection_radius * radius_scale
            for source in range(count):
                order = np.argsort(distances[source])
                linked = 0
                for target in order:
                    if source == target or distances[source, target] > radius:
                        continue
                    if self.map.line_is_free(
                        self.positions[source], self.positions[target], margin=0.08
                    ):
                        adjacency[source, target] = True
                        adjacency[target, source] = True
                        linked += 1
                        if linked >= max_neighbors:
                            break

            components, _ = connected_components(
                csr_matrix(adjacency), directed=False
            )
            if components == 1:
                return adjacency

        raise RuntimeError(
            "Automatic landmark graph is disconnected; increase max_landmarks "
            "or connection_radius"
        )

    def _build_directed_actions(self):
        self.directed_edges = []
        self.action_by_edge = {}
        self.outgoing_actions = [[] for _ in range(self.num_nodes)]
        for source in range(self.num_nodes):
            for target in np.flatnonzero(self.adjacency[source]):
                action_id = len(self.directed_edges)
                edge = (source, int(target))
                self.directed_edges.append(edge)
                self.action_by_edge[edge] = action_id
                self.outgoing_actions[source].append(action_id)
        self.directed_edges = np.asarray(self.directed_edges, dtype=np.int64)

    def action_target(self, action_id):
        return int(self.directed_edges[action_id, 1])

    def nearest_visible_node(self, point):
        return self.visible_nodes(point, count=1)[0]

    def visible_nodes(self, point, count=8):
        point = np.asarray(point, dtype=np.float64)
        distances = np.linalg.norm(self.positions - point[None, :], axis=1)
        visible = []
        for node in np.argsort(distances):
            if self.map.line_is_free(point, self.positions[node], margin=0.05):
                visible.append(int(node))
                if len(visible) >= count:
                    return visible
        if visible:
            return visible
        raise RuntimeError("No landmark is visible from the requested position")

    def path_length(self, node_path):
        if len(node_path) < 2:
            return 0.0
        points = self.positions[np.asarray(node_path, dtype=np.int64)]
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def waypoints(self, start, goal, node_path):
        points = [np.asarray(start, dtype=np.float64)]
        points.extend(self.positions[node].copy() for node in node_path)
        points.append(np.asarray(goal, dtype=np.float64))
        deduplicated = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - deduplicated[-1]) > 1e-8:
                deduplicated.append(point)
        return np.asarray(deduplicated)
