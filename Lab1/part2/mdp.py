import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, List

class GridWorld5x5Env(gym.Env):
    """
    5x5 GridWorld with:
    - start (0,0)
    - goal (4,4) terminal
    - blocks (2,1), (2,3)
    - step reward -1, goal reward +10
    - stochastic: 0.8 intended, 0.1 left-perp, 0.1 right-perp
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # action meanings (match your dx/dy idea)
    # 0: Up, 1: Left, 2: Down, 3: Right
    ACTIONS = ["U", "L", "D", "R"]
    DX = np.array([0, -1, 0, 1], dtype=np.int32)
    DY = np.array([1, 0, -1, 0], dtype=np.int32)

    def __init__(
        self,
        gamma: float = 0.9,
        stochastic: bool = True,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.W = self.H = 5
        self.gamma = gamma
        self.stochastic = stochastic
        self.render_mode = render_mode

        self.start = (0, 0)
        self.goal = (4, 4)
        self.blocks = {(2, 1), (2, 3)}

        # RNG (Gymnasium style)
        self.np_random = None
        self._seed = seed

        # Observation space
        self.observation_space: spaces.Discrete = spaces.Discrete(self.W * self.H)

        self.action_space: spaces.Discrete = spaces.Discrete(4)

        # internal state (x,y)
        self.location = np.array(self.start, dtype=np.int32)

        # Precompute valid states list for planning convenience
        self.states = [(x, y) for x in range(self.W) for y in range(self.H) if (x, y) not in self.blocks]

    # ---------- Helpers ----------
    def _state_to_id(self, s: Tuple[int, int]) -> int:
        x, y = s
        return y * self.W + x

    def _id_to_state(self, sid: int) -> Tuple[int, int]:
        x = sid % self.W
        y = sid // self.W
        return (x, y)

    def _get_obs(self):
        return self._state_to_id((int(self.location[0]), int(self.location[1])))

    def _get_info(self) -> Dict:
        # Manhattan distance to goal
        dist = abs(int(self.location[0]) - self.goal[0]) + abs(int(self.location[1]) - self.goal[1])
        return {"distance": dist}

    def is_terminal_state(self, s: Tuple[int, int]) -> bool:
        return s == self.goal

    def _left_perp(self, a: int) -> int:
        return (a + 1) % 4

    def _right_perp(self, a: int) -> int:
        return (a + 3) % 4

    def _next_state(self, s: Tuple[int, int], a: int) -> Tuple[int, int]:
        """Deterministic movement with walls/blocks."""
        x, y = s
        nx = x + int(self.DX[a])
        ny = y + int(self.DY[a])

        # off-grid => stay
        if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
            return (x, y)
        # into block => stay
        if (nx, ny) in self.blocks:
            return (x, y)
        return (nx, ny)

    def _reward_and_done(self, s2: Tuple[int, int]) -> Tuple[int, bool]:
        if s2 == self.goal:
            return 10, True
        return -1, False

    # ---------- Gymnasium API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # create RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed if seed is not None else self._seed)

        self.location = np.array(self.start, dtype=np.int32)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        s = (int(self.location[0]), int(self.location[1]))

        # If already terminal: remain terminal (common convention)
        if self.is_terminal_state(s):
            return self._get_obs(), 0, True, False, self._get_info()

        # choose executed action
        if self.stochastic:
            p = float(self.np_random.random())
            if p < 0.8:
                a_exec = action
            elif p < 0.9:
                a_exec = self._left_perp(action)
            else:
                a_exec = self._right_perp(action)
        else:
            a_exec = action

        s2 = self._next_state(s, a_exec)
        r, terminated = self._reward_and_done(s2)

        # update internal state
        self.location = np.array(s2, dtype=np.int32)

        truncated = False
        obs = self._get_obs()
        info = self._get_info()
        return obs, r, terminated, truncated, info

    # ---------- Model access for planning (Task 1) ----------
    def transitions(self, s: Tuple[int, int], a: int) -> List[Tuple[float, Tuple[int, int], int, bool]]:
        """
        Returns list of (prob, s2, r, done).
        Works for deterministic or stochastic modes.
        """
        if self.is_terminal_state(s):
            return [(1.0, s, 0, True)]

        if not self.stochastic:
            s2 = self._next_state(s, a)
            r, done = self._reward_and_done(s2)
            return [(1.0, s2, r, done)]

        dist = []
        for prob, a_exec in [(0.8, a), (0.1, self._left_perp(a)), (0.1, self._right_perp(a))]:
            s2 = self._next_state(s, a_exec)
            r, done = self._reward_and_done(s2)
            dist.append((prob, s2, r, done))
        return dist

    # For rendering purpose only
    def render(self):
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        for (bx, by) in self.blocks:
            grid[by][bx] = "#"
        ax, ay = int(self.location[0]), int(self.location[1])
        gx, gy = self.goal
        grid[gy][gx] = "G"
        grid[ay][ax] = "A"
        print("\n".join(" ".join(row) for row in reversed(grid)))
        print()