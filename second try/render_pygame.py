import pygame
import math
import numpy as np
from parking_env import ParkingEnv


class ParkingRenderer:
    def __init__(self, env: ParkingEnv, scale=40):
        pygame.init()
        self.env = env
        self.scale = scale

        size = int(env.area_size * 2 * scale)
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Self-Parking Visualization")

        self.clock = pygame.time.Clock()

    def world_to_screen(self, x, y):
        """Convert env coordinates (center=0,0) to pygame coordinates."""
        size = int(self.env.area_size * 2 * self.scale)
        screen_x = int(size / 2 + x * self.scale)
        screen_y = int(size / 2 - y * self.scale)
        return screen_x, screen_y

    def draw_car(self, x, y, theta):
        length = 0.6
        width = 0.35

        corners = np.array([
            [ length/2,  0],
            [-length/2,  width/2],
            [-length/2, -width/2],
        ])

        rot = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)],
        ])

        rotated = (rot @ corners.T).T + np.array([x, y])

        pts = [self.world_to_screen(px, py) for px, py in rotated]
        pygame.draw.polygon(self.screen, (80, 150, 255), pts)

    def draw_goal(self):
        gx, gy = self.env.goal_pos
        sx, sy = self.world_to_screen(gx, gy)
        pygame.draw.circle(self.screen, (0, 255, 0), (sx, sy), 8)

    def render(self, obs):
        x, y, vx, vy, theta, omega, dx, dy, terr = obs

        self.screen.fill((30, 30, 30))

        self.draw_goal()
        self.draw_car(x, y, theta)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()


# ----------------------------------------------------------
# Run viewer with a random or trained policy
# ----------------------------------------------------------
def run_viewer(model=None):
    env = ParkingEnv()
    obs, info = env.reset()

    viewer = ParkingRenderer(env)

    running = True
    while running:

        # Quit window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Choose action
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)

        viewer.render(obs)

        if done or truncated:
            obs, info = env.reset()

    viewer.close()


if __name__ == "__main__":
    run_viewer()
