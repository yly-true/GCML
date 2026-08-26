import heapq

import numpy as np
from scipy.sparse.csgraph import laplacian


class LandmarkGCML:
    """Lightweight Q/V/W GCML operating on a sparse landmark graph."""

    def __init__(
        self,
        graph,
        latent_dim=24,
        eta_q=0.002,
        eta_v=0.04,
        eta_w=0.01,
        noise_std=0.18,
        seed=0,
    ):
        self.graph = graph
        self.rng = np.random.default_rng(seed)
        self.latent_dim = min(int(latent_dim), graph.num_nodes - 1)
        self.eta_q = float(eta_q)
        self.eta_v = float(eta_v)
        self.eta_w = float(eta_w)
        self.noise_std = float(noise_std)

        graph_laplacian = laplacian(graph.adjacency.astype(float), normed=True)
        _, eigenvectors = np.linalg.eigh(graph_laplacian)
        self.Q = eigenvectors[:, 1 : self.latent_dim + 1].T.copy()
        self.Q += self.rng.normal(0.0, 0.01, size=self.Q.shape)
        self._whiten_q()

        self.Q_initial = self.Q.copy()
        self.V = self.rng.normal(
            0.0, 0.05, size=(self.latent_dim, graph.num_actions)
        )
        self.W = self.rng.normal(
            0.0, 0.05, size=(graph.num_actions, self.latent_dim)
        )
        self.V_initial = self.V.copy()
        self.W_initial = self.W.copy()

    def _whiten_q(self):
        self.Q -= np.mean(self.Q, axis=1, keepdims=True)
        covariance = self.Q @ self.Q.T / self.Q.shape[1]
        values, vectors = np.linalg.eigh(covariance)
        inverse_root = vectors @ np.diag(1.0 / np.sqrt(values + 1e-6)) @ vectors.T
        self.Q = inverse_root @ self.Q

    def train(self, epochs=120):
        edges = self.graph.directed_edges
        action_ids = np.arange(self.graph.num_actions)

        for _ in range(int(epochs)):
            for action_id in self.rng.permutation(action_ids):
                source, target = edges[action_id]
                source = int(source)
                target = int(target)
                s_source = self.Q[:, source].copy()
                s_target = self.Q[:, target].copy()
                state_diff = s_target - s_source
                error = state_diff - self.V[:, action_id]

                self.V[:, action_id] += self.eta_v * error
                # Paper equation (13): only the current observation column
                # receives (predicted next state - observed next state).
                self.Q[:, source] -= self.eta_q * error

                # Paper equation (14), with a one-hot edge action.
                self.W[action_id] += self.eta_w * state_diff
            self._whiten_q()

        # Let V catch up with the final Q without changing the embedding.
        for _ in range(30):
            for action_id, (source, target) in enumerate(edges):
                state_diff = self.Q[:, target] - self.Q[:, source]
                self.V[:, action_id] += self.eta_v * (
                    state_diff - self.V[:, action_id]
                )

    def transition_rmse(self):
        errors = []
        for action_id, (source, target) in enumerate(self.graph.directed_edges):
            error = (
                self.Q[:, target]
                - self.Q[:, source]
                - self.V[:, action_id]
            )
            errors.append(float(error @ error))
        return float(np.sqrt(np.mean(errors)))

    def diagnostics(self):
        return {
            "Q_change": float(np.linalg.norm(self.Q - self.Q_initial)),
            "V_change": float(np.linalg.norm(self.V - self.V_initial)),
            "W_change": float(np.linalg.norm(self.W - self.W_initial)),
            "transition_rmse": self.transition_rmse(),
            "finite": all(
                np.all(np.isfinite(parameter))
                for parameter in (self.Q, self.V, self.W)
            ),
            "parameter_count": int(self.Q.size + self.V.size + self.W.size),
        }

    def _sample_path(self, start_node, goal_node, horizon, noise_scale):
        current = int(start_node)
        path = [current]
        actions = []
        imagined_state = self.Q[:, current].copy()
        goal_state = self.Q[:, goal_node]

        for _ in range(horizon):
            if current == goal_node:
                break
            outgoing = np.asarray(
                self.graph.outgoing_actions[current], dtype=np.int64
            )
            if len(outgoing) == 0:
                break

            # Equations (16), (18), (19): utility, masked noisy eligibility,
            # and winner-take-all. The graph supplies the affordance mask.
            utility = self.W @ (goal_state - imagined_state)
            utility /= max(float(np.linalg.norm(utility)), 1e-9)
            eligibility = utility[outgoing]
            eligibility += self.rng.normal(
                0.0,
                self.noise_std * float(noise_scale),
                size=len(outgoing),
            )
            local_index = int(np.argmax(eligibility))
            action_id = int(outgoing[local_index])
            current = self.graph.action_target(action_id)
            actions.append(action_id)
            path.append(current)
            # Equation (20): bootstrap without observing/snap-to Q(o_{t+1}).
            imagined_state += self.V[:, action_id]

        path = self._erase_loops(path)
        actions = [
            self.graph.action_by_edge[(source, target)]
            for source, target in zip(path[:-1], path[1:])
        ]
        return path, actions

    @staticmethod
    def _erase_loops(path):
        simplified = []
        location = {}
        for node in path:
            if node in location:
                keep_through = location[node]
                for removed in simplified[keep_through + 1 :]:
                    location.pop(removed, None)
                simplified = simplified[: keep_through + 1]
            else:
                location[node] = len(simplified)
                simplified.append(node)
        return simplified

    def _path_score(self, path, goal_node):
        success = path[-1] == goal_node
        length = self.graph.path_length(path)
        revisits = len(path) - len(set(path))
        if success:
            return (0, length + 1.5 * revisits)
        remaining = self.graph.shortest_distances[path[-1], goal_node]
        return (1, remaining + 0.08 * length + 2.0 * revisits)

    def _shortest_path_fallback(self, start_node, goal_node):
        queue = [(0.0, int(start_node))]
        parent = {int(start_node): None}
        cost = {int(start_node): 0.0}
        while queue:
            current_cost, node = heapq.heappop(queue)
            if node == goal_node:
                break
            if current_cost != cost[node]:
                continue
            for action_id in self.graph.outgoing_actions[node]:
                target = self.graph.action_target(action_id)
                edge_cost = float(
                    np.linalg.norm(
                        self.graph.positions[target] - self.graph.positions[node]
                    )
                )
                candidate = current_cost + edge_cost
                if candidate < cost.get(target, np.inf):
                    cost[target] = candidate
                    parent[target] = node
                    heapq.heappush(queue, (candidate, target))
        if goal_node not in parent:
            raise RuntimeError("Landmark graph has no route to the goal")
        path = []
        node = int(goal_node)
        while node is not None:
            path.append(node)
            node = parent[node]
        return path[::-1]

    def rollout(self, start_node, goal_node, candidates=384, horizon=None):
        if horizon is None:
            horizon = max(32, 2 * self.graph.num_nodes)
        best_path = None
        best_actions = None
        best_key = None

        for index in range(int(candidates)):
            noise_scale = 0.55 + 1.15 * (index / max(candidates - 1, 1))
            path, actions = self._sample_path(
                start_node, goal_node, horizon, noise_scale
            )
            key = self._path_score(path, goal_node)
            if best_key is None or key < best_key:
                best_key = key
                best_path = path
                best_actions = actions

        fallback_used = best_path[-1] != goal_node
        if fallback_used:
            best_path = self._shortest_path_fallback(start_node, goal_node)
            best_actions = [
                self.graph.action_by_edge[(source, target)]
                for source, target in zip(best_path[:-1], best_path[1:])
            ]
            best_key = self._path_score(best_path, goal_node)

        return {
            "success": best_path[-1] == goal_node,
            "nodes": best_path,
            "actions": best_actions,
            "score": best_key[1],
            "candidate_count": int(candidates),
            "fallback_used": fallback_used,
        }
