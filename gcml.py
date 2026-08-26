import numpy as np


class GCML:
    """
    Generic 2026-style GCML core:

        s_t = Q o_t
        s_hat_{t+1} = s_t + V a_t

        dW = eta_w * (a_t - W (s_{t+1} - s_t))
                         (s_{t+1} - s_t)^T
        dG = eta_g * (g_t - G s_t) s_t^T

    imagination:

        u_t = W (s_goal - s_hat_t)
        e_t = g_hat_t * (u_t + epsilon)
        a_t = WTA(e_t)
        s_hat_{t+1} = s_hat_t + V a_t
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_dim=64,
        eta_q=0.10,
        eta_v=0.01,
        eta_w=0.01,
        eta_g=0.01,
        noise_std=0.10,
        seed=0,
    ):
        rng = np.random.default_rng(seed)

        self.Q = rng.normal(0.0, 1.0, size=(latent_dim, obs_dim))
        self.V = rng.normal(0.0, 0.1, size=(latent_dim, action_dim))
        self.W = rng.normal(0.0, 0.1, size=(action_dim, latent_dim))
        self.G = rng.normal(0.0, 0.1, size=(action_dim, latent_dim))

        self.eta_q = eta_q
        self.eta_v = eta_v
        self.eta_w = eta_w
        self.eta_g = eta_g
        self.noise_std = noise_std
        self.rng = rng
        self.affordance_threshold = 0.5

    def encode(self, o):
        return self.Q @ o

    def learn_transition(self, o_t, a_t, o_next, g_t):
        # states before parameter update
        s_t = self.Q @ o_t
        s_next = self.Q @ o_next

        # Eq. 10: predicted next state
        s_hat_next = s_t + self.V @ a_t

        # Eq. 12: V update
        prediction_error = s_next - s_hat_next
        self.V += self.eta_v * np.outer(prediction_error, a_t)

        # Semi-gradient of Eq. 11 with the next-state embedding held fixed.
        # The printed sign in Eq. 13 is an anti-gradient if it is applied to
        # o_t with an in-place += update (which is what the original framework
        # did); using prediction_error moves Q o_t toward Q o_next - V a_t.
        self.Q += self.eta_q * np.outer(prediction_error, o_t)

        # Error-corrected inverse-model update used by the released code. The
        # raw Hebbian form grows without bound on long random walks.
        state_diff = s_next - s_t
        action_error = a_t - self.W @ state_diff
        state_diff_energy = float(state_diff @ state_diff)
        self.W += (
            self.eta_w
            * np.outer(action_error, state_diff)
            / max(state_diff_energy, 1e-12)
        )

        # Eq. 17: affordance G update
        g_hat = self.G @ s_t
        state_energy = float(s_t @ s_t)
        self.G += (
            self.eta_g
            * np.outer(g_t - g_hat, s_t)
            / max(state_energy, 1e-12)
        )

    def choose_imagined_action(self, s_hat, s_goal, action_mask=None):
        # Eq. 16
        u = self.W @ (s_goal - s_hat)

        # Methods: noise is added to a normalized unit-length utility vector.
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-12:
            u = u / u_norm

        if action_mask is None:
            # Generic graph mode: use the learned imagined affordance.
            g_hat = np.clip(self.G @ s_hat, 0.0, 1.0)
        else:
            # Spatial mode: walls are exogenous constraints, analogous to the
            # obstacle/barrier signal used in the paper's spatial experiment.
            g_hat = np.asarray(action_mask, dtype=np.float64)
            if g_hat.shape != u.shape:
                raise ValueError("action_mask must have one value per action")

        # Eq. 18
        eps = self.rng.normal(0.0, self.noise_std, size=u.shape)
        eligibility = g_hat * (u + eps)

        # Affordance is a binary veto in the paper. Hard masking also prevents
        # an infeasible zero from beating a feasible negative noisy utility.
        valid = g_hat >= self.affordance_threshold
        if np.any(valid):
            eligibility = np.where(valid, eligibility, -np.inf)
        else:
            best_affordance = np.max(g_hat)
            fallback = np.isclose(g_hat, best_affordance)
            eligibility = np.where(fallback, u + eps, -np.inf)

        # Eq. 19: WTA
        return int(np.argmax(eligibility))

    def rollout(
        self,
        o_start,
        o_goal,
        horizon=100,
        goal_state_id=None,
        start_state_id=None,
        action_mask_fn=None,
        transition_fn=None,
    ):
        s_hat = self.encode(o_start)
        s_goal = self.encode(o_goal)
        state_id = start_state_id

        latent_path = [s_hat.copy()]
        actions = []

        for _ in range(horizon):
            action_mask = (
                action_mask_fn(state_id)
                if action_mask_fn is not None and state_id is not None
                else None
            )
            action_id = self.choose_imagined_action(
                s_hat, s_goal, action_mask=action_mask
            )

            a = np.zeros(self.V.shape[1], dtype=np.float64)
            a[action_id] = 1.0

            # Eq. 20: bootstrapping
            s_hat = s_hat + self.V @ a

            actions.append(action_id)
            latent_path.append(s_hat.copy())

            if transition_fn is not None and state_id is not None:
                state_id = transition_fn(state_id, action_id)

            if (
                goal_state_id is not None
                and (
                    state_id == goal_state_id
                    if state_id is not None
                    else self.decode_state(s_hat) == goal_state_id
                )
            ):
                break

        return actions, np.asarray(latent_path)

    def transition_error(self, o_t, a_t, o_next):
        """Return ||Q o_next - (Q o_t + V a_t)|| for diagnostics."""
        prediction = self.encode(o_t) + self.V @ a_t
        return float(np.linalg.norm(self.encode(o_next) - prediction))

    def decode_state(self, s_hat):
        """
        Only for visualization/debugging:
        nearest learned state embedding among columns of Q.
        """
        d = np.linalg.norm(self.Q.T - s_hat[None, :], axis=1)
        return int(np.argmin(d))
