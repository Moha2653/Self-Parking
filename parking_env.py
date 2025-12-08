# parking_env.py
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# --- Layout params (based on your game) ---
W, H = 300, 300
SPOT_W, SPOT_H = 65, 35
MARGIN_TOP = 25
MARGIN_SIDE = 20
FPS = 30

class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode: str = None, seed: int | None = None, noise: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.noise = noise
        self.seed(seed)

        # Discrete actions: 0 noop,1 gas,2 brake,3 left,4 right,5 reverse
        self.action_space = spaces.Discrete(6)

        # Observation: x, y, vx, vy, theta (deg), omega (deg/s), dx, dy, theta_error (deg)
        obs_high = np.array([W, H, 10.0, 10.0, 360.0, 360.0, W, H, 180.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Internal state
        self.spots = self._build_parking_spots()
        self.screen = None
        self.clock = None
        self.font = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _build_parking_spots(self):
        spots = []
        for row in range(6):
            y = MARGIN_TOP + row * (SPOT_H + 5)
            left = pygame.Rect(MARGIN_SIDE, y, SPOT_W, SPOT_H)
            right = pygame.Rect(W - MARGIN_SIDE - SPOT_W, y, SPOT_W, SPOT_H)
            spots.extend([left, right])
        return spots

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        # start position near bottom-center
        self.x = W // 2 + self.np_random.uniform(-10, 10)
        self.y = H - 80 + self.np_random.uniform(-5, 5)
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 90.0 + self.np_random.uniform(-15, 15)  # degrees
        self.omega = 0.0
        self.steps = 0
        # choose random target and obstacles
        self.spots = self._build_parking_spots()
        self.target_idx = int(self.np_random.integers(0, len(self.spots)))
        self.target_rect = self.spots[self.target_idx]
        self.target_pos = (self.target_rect.centerx, self.target_rect.centery)
        self.obstacles = [r for i, r in enumerate(self.spots) if i != self.target_idx]
        return self._get_obs(), {}

    def _angle_diff(self, a_deg, b_deg):
        d = (a_deg - b_deg + 180) % 360 - 180
        return d

    def _get_obs(self):
        dx = self.target_pos[0] - self.x
        dy = self.target_pos[1] - self.y
        theta_err = self._angle_diff(0.0, self.theta)  # target angle = 0 deg (park aligned)
        return np.array([self.x, self.y, self.vx, self.vy, self.theta, self.omega, dx, dy, theta_err],
                        dtype=np.float32)

    def _apply_action(self, action):
        # simple physics discretized per step (FPS-based constants)
        accel = 6.0 / FPS
        turn = 100.0 / FPS  # degrees per step
        if action == 1:  # gas
            speed = math.hypot(self.vx, self.vy) + accel
        elif action == 2:  # brake
            speed = math.hypot(self.vx, self.vy) - accel * 1.2
        elif action == 5:  # reverse (we treat as negative accel)
            speed = math.hypot(self.vx, self.vy) - accel * 0.7
        else:
            # friction
            speed = math.hypot(self.vx, self.vy) * 0.9

        # clamp speed
        speed = max(min(speed, 6.0), -3.0)

        # steering
        if action == 3:
            self.theta += turn
        elif action == 4:
            self.theta -= turn

        # convert speed + theta to vx, vy
        rad = math.radians(self.theta)
        self.vx = speed * math.cos(rad)
        self.vy = -speed * math.sin(rad)  # screen y inverted

        # update pos
        self.x += self.vx
        self.y += self.vy

        # omega approx
        self.omega = 0.0  # not modelling continuous angular velocity precisely

        # optional environment noise
        if self.noise:
            # lateral wind
            wind = self.np_random.normal(0, 0.2)
            self.x += wind

    def _check_collision(self):
        # out of bounds
        if not (0 < self.x < W and 0 < self.y < H):
            return True, "wall"
        # simple hitbox
        hit = pygame.Rect(self.x - 10, self.y - 10, 20, 20)
        for obs in self.obstacles:
            if hit.colliderect(obs):
                return True, "car"
        return False, None

    def _check_parked(self):
        dist = math.hypot(self.target_pos[0] - self.x, self.target_pos[1] - self.y)
        theta_err = abs(self._angle_diff(self.theta, 0.0))
        speed = math.hypot(self.vx, self.vy)
        if dist < 12 and theta_err < 25 and abs(speed) < 0.4:
            # inside target rect?
            if self.target_rect.collidepoint(self.x, self.y):
                return True
        return False

    def _compute_reward(self, collided: bool, crash_type: str | None, parked: bool):
        if parked:
            return 200.0
        if collided:
            return -100.0
        # shaping: negative distance + small angular penalty + small speed penalty
        dist = math.hypot(self.target_pos[0] - self.x, self.target_pos[1] - self.y)
        ang_err = abs(self._angle_diff(self.theta, 0.0)) / 180.0
        speed = math.hypot(self.vx, self.vy)
        reward = -dist * 0.5 - ang_err * 5.0 - speed * 0.5
        return reward

    def step(self, action):
        self.steps += 1
        self._apply_action(int(action))
        collided, crash_type = self._check_collision()
        parked = self._check_parked()
        terminated = False
        info = {"collision": collided, "crash_type": crash_type, "success": parked}
        if collided:
            terminated = True
        if parked:
            terminated = True
        truncated = self.steps >= 400
        reward = self._compute_reward(collided, crash_type, parked)
        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------------- Pygame rendering ----------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("ParkingEnv (RL)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14)
        self.screen.fill((50, 55, 60))
        # draw spots
        for i, rect in enumerate(self.spots):
            if i == self.target_idx:
                pygame.draw.rect(self.screen, (60, 200, 60), rect.inflate(-8, -8))
            else:
                pygame.draw.rect(self.screen, (120, 120, 140), rect.inflate(-10, -10))
        # draw agent (rotated rectangle)
        car_w, car_h = 42, 22
        surf = pygame.Surface((car_w, car_h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (200, 60, 60), (0, 0, car_w, car_h), border_radius=3)
        rot = pygame.transform.rotate(surf, self.theta)
        r = rot.get_rect(center=(self.x, self.y))
        self.screen.blit(rot, r)
        # HUD
        txt = f"Step:{self.steps}  x:{int(self.x)} y:{int(self.y)} speed:{math.hypot(self.vx,self.vy):.2f}"
        self.screen.blit(self.font.render(txt, True, (255,255,255)), (5, H-22))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
