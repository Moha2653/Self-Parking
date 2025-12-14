import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
 
class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
 
    def __init__(self, render_mode=None, robustness=False):
        super(ParkingEnv, self).__init__()
       
        # Variables
        self.W, self.H = 400, 400
        self.SPOT_W, self.SPOT_H = 65, 35
        self.MARGIN = 20
        self.render_mode = render_mode
        self.robustness = robustness
       
        self.max_steps = 800
        self.car_w, self.car_h = 44, 22
       
        self.action_space = spaces.Discrete(5)
 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
 
        self.window = None
        self.clock = None
       
        self._init_geometry()
 
    def _init_geometry(self):
        self.spots = []
        for col_x in [self.MARGIN, self.W - self.MARGIN - self.SPOT_W]:
            for row in range(6):
                y = 40 + row * (self.SPOT_H + 10)
                self.spots.append(pygame.Rect(col_x, y, self.SPOT_W, self.SPOT_H))
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
       
        self.target_idx = random.randint(0, len(self.spots) - 1)
        self.target_rect = self.spots[self.target_idx]
        self.obstacles = [s for i, s in enumerate(self.spots) if i != self.target_idx]
       
        #Trcuk set up
        start_x = self.W // 2 + random.randint(-40, 40)
        start_y = self.H - 60
        self.pos = [float(start_x), float(start_y)]
        self.speed = 0.0
        self.orientation = 90.0
        self.ang_vel = 0.0
       
        self.prev_pos = (self.pos[0], self.pos[1])
        self.spin_timer = 0
        self.spin_angle_accum = 0.0
 
        self.steps = 0
        self.prev_dist = math.hypot(self.target_rect.centerx - self.pos[0],
                                    self.target_rect.centery - self.pos[1])
 
        # Random friction
        if self.robustness:
            self.friction = np.random.uniform(0.85, 0.96)
        else:
            self.friction = 0.90
 
        return self._get_obs(), {}
 
    def _get_obs(self):
        rad = math.radians(self.orientation)
        vx = self.speed * math.cos(rad)
        vy = -self.speed * math.sin(rad)
        dx = self.target_rect.centerx - self.pos[0]
        dy = self.target_rect.centery - self.pos[1]
        obs = np.array([
            self.pos[0] / self.W,
            self.pos[1] / self.H,
            vx / 5.0,
            vy / 5.0,
            math.cos(rad),
            math.sin(rad),
            self.ang_vel,
            dx / self.W,
            dy / self.H
        ], dtype=np.float32)
 
        # Robustness (Sensor Noise)
        if self.robustness:
            noise = np.random.normal(0, 0.02, size=obs.shape)
            obs += noise
 
        return obs
 
    def step(self, action):
        self.steps += 1
        dt = 1/60.0
       
        # --- 1. Physics ---
        if action == 1: self.speed += 6 * dt        # Accelerate
        elif action == 2: self.speed -= 6 * dt      # Reverse
        else: self.speed *= self.friction          
 
        turn_factor = 130 if abs(self.speed) < 1.0 else 90
        if action == 3: self.ang_vel = turn_factor * dt
        elif action == 4: self.ang_vel = -turn_factor * dt
        else: self.ang_vel = 0
 
        self.orientation = (self.orientation + self.ang_vel) % 360
        rad = math.radians(self.orientation)
        self.pos[0] += self.speed * math.cos(rad)
        self.pos[1] -= self.speed * math.sin(rad)
 
        # --- 2. Reward Logic ---
        reward = 0
        terminated = False
        truncated = False
        
        curr_dist = math.hypot(
            self.target_rect.centerx - self.pos[0],
            self.target_rect.centery - self.pos[1]
        )

        # Rewarding
        reward += (self.prev_dist - curr_dist) * 0.5
        self.prev_dist = curr_dist

        ang_err = (self.orientation % 180)
        if ang_err > 90: 
            ang_err = 180 - ang_err  # error absoluto
        reward -= (ang_err / 90) * 0.1   # normalizado

        if curr_dist < 50 and abs(self.speed) > 1.5:
            reward -= 1.0

        reward -= 0.01

        if curr_dist < 50 and abs(self.speed) < 0.5:
            reward += 1.0

        if abs(self.speed) < 0.05:
            reward -= 0.05

        # Spinning penalty
        linear_movement = math.hypot(
            self.pos[0] - self.prev_pos[0],
            self.pos[1] - self.prev_pos[1]
        )
        if linear_movement < 0.5:
            self.spin_timer += 1
            self.spin_angle_accum += abs(self.ang_vel * 60)
        else:
            self.spin_timer = 0
            self.spin_angle_accum = 0.0

        if self.spin_timer > 30 or self.spin_angle_accum > 360:
            reward -= 0.3

        self.prev_pos = (self.pos[0], self.pos[1])

        #Crashing
        crashed = False
        player_rect = pygame.Rect(0, 0, self.car_w, self.car_h)
        player_rect.center = (int(self.pos[0]), int(self.pos[1]))
        hitbox = player_rect.inflate(-10, -10)

        if not (0 < self.pos[0] < self.W and 0 < self.pos[1] < self.H):
            crashed = True

        for obs in self.obstacles:
            if hitbox.colliderect(obs):
                crashed = True

        if crashed:
            reward = -50
            terminated = True

        #Wininng
        is_close = curr_dist < 15
        is_slow = abs(self.speed) < 1.0
        is_aligned = ang_err < 15

        if is_close and is_slow and is_aligned:
            if self.target_rect.collidepoint(self.pos[0], self.pos[1]):
                reward += 100
                terminated = True

        # --- (9) TIMEOUT ---
        if self.steps >= self.max_steps:
            truncated = True
 
        if self.render_mode == "human":
            self.render()
 
        return self._get_obs(), reward, terminated, truncated, {}
 
    def render(self):
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.W, self.H))
            else:
                # rgb_array → surface invisible
                self.window = pygame.Surface((self.W, self.H))

            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14, bold=True)

        canvas = pygame.Surface((self.W, self.H))
        canvas.fill((50, 55, 60))

        # Draw Target & Obstacles
        for i, rect in enumerate(self.spots):
            if i == self.target_idx:
                pygame.draw.rect(canvas, (50, 150, 50), rect, 2)
                lbl = self.font.render("PARK", True, (100, 200, 100))
                canvas.blit(lbl, (rect.x + 10, rect.y + 10))
            else:
                color = (100 + (i*20)%150, 100, 150)
                pygame.draw.rect(canvas, color, rect.inflate(-5,-5), border_radius=4)

        # Draw car
        truck_surf = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        pygame.draw.rect(truck_surf, (160, 30, 30), (0, 0, 20, 22)) 
        pygame.draw.rect(truck_surf, (100, 20, 20), (2, 2, 12, 18))

        pygame.draw.rect(truck_surf, (220, 40, 40), (20, 0, 28, 22), border_radius=3)
        pygame.draw.rect(truck_surf, (180, 30, 30), (22, 2, 20, 20)) 
        pygame.draw.rect(truck_surf, (100, 200, 255), (self.car_w - 12, 2, 7, self.car_h - 4), border_radius=2)

        pygame.draw.circle(truck_surf, (255, 255, 200), (self.car_w-1, 4), 2)
        pygame.draw.circle(truck_surf, (255, 255, 200), (self.car_w-1, self.car_h-4), 2)

        rot_truck = pygame.transform.rotate(truck_surf, self.orientation)
        truck_rect = rot_truck.get_rect(center=(self.pos[0], self.pos[1]))
        canvas.blit(rot_truck, truck_rect)

        # Human mode → visualizar en pantalla
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()

        self.clock.tick(self.metadata["render_fps"])

        # rgb_array → devolver frame como numpy array
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(canvas).swapaxes(0, 1)

    def close(self):
        if self.window is not None:
            pygame.quit()