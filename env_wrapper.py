import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid environments
import numpy as np


class FourRoomsDiscrete:
    """
    Use the official MiniGrid-FourRooms-v0 map, but expose a GCML-friendly
    discrete MDP:

      observation: one-hot(cell_id)
      action:       one-hot([up, down, left, right])

    We intentionally use absolute moves rather than MiniGrid's native
    left/right/forward controls, because the GCML forward model assumes
    a fixed action displacement V a.
    """

    ACTIONS = np.array([
        [0, -1],   # up
        [0,  1],   # down
        [-1, 0],   # left
        [1,  0],   # right
    ], dtype=np.int64)

    def __init__(self, seed=0):
        self.env = gym.make("MiniGrid-FourRooms-v0")
        self.env.reset(seed=seed)
        base = self.env.unwrapped

        self.width = base.width
        self.height = base.height

        self.free_cells = []
        self.goal_pos = None

        for y in range(self.height):
            for x in range(self.width):
                cell = base.grid.get(x, y)
                if cell is None or getattr(cell, "type", None) != "wall":
                    self.free_cells.append((x, y))
                if cell is not None and getattr(cell, "type", None) == "goal":
                    self.goal_pos = (x, y)

        if self.goal_pos is None:
            raise RuntimeError("MiniGrid FourRooms did not generate a goal cell")

        self.state_to_id = {p: i for i, p in enumerate(self.free_cells)}
        self.id_to_state = {i: p for i, p in enumerate(self.free_cells)}

        self.num_states = len(self.free_cells)
        self.num_actions = 4

        self.start_pos = tuple(int(v) for v in base.agent_pos)
        self.start_id = self.state_to_id[self.start_pos]
        self.goal_id = self.state_to_id[self.goal_pos]

    def obs(self, state_id):
        o = np.zeros(self.num_states, dtype=np.float64)
        o[state_id] = 1.0
        return o

    def action_one_hot(self, action_id):
        a = np.zeros(self.num_actions, dtype=np.float64)
        a[action_id] = 1.0
        return a

    def transition(self, state_id, action_id):
        x, y = self.id_to_state[state_id]
        dx, dy = self.ACTIONS[action_id]
        nxt = (x + int(dx), y + int(dy))

        if nxt not in self.state_to_id:
            return state_id

        return self.state_to_id[nxt]

    def affordance(self, state_id):
        g = np.zeros(self.num_actions, dtype=np.float64)
        for a in range(self.num_actions):
            nxt = self.transition(state_id, a)
            if nxt != state_id:
                g[a] = 1.0
        return g
