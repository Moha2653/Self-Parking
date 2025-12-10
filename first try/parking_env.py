import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

class ParkingEnv(gym.Env):
  """Entorno de estacionamiento para un vehículo autónomo.
  Termina cuando:
  - El auto se estaciona bien.
  - El auto sale del área.
  - Se alcanza el máximo de steps.
  """
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

  def __init__(
      self,
      render_mode: Optional[str] = None,
      dt: float = 0.1,
      wheelbase: float = 0.5,
      max_speed: float = 3.0,
      max_steering: float = math.radians(30),
      area_size: float = 10.0,
      goal_pos: Tuple[float, float] = (0.0, 0.0),
      goal_theta: float = 0.0,
      max_steps: int = 200,
      seed: Optional[int] = None):
    super().__init__()
    self.dt = dt
    self.wheelbase = wheelbase
    self.max_speed = max_speed
    self.max_steering = max_steering
    self.area_size = area_size
    self.goal_pos = np.array(goal_pos, dtype=np.float32)
    self.goal_theta = float(goal_theta)
    self.max_steps = max_steps

    self.action_space = spaces.Box(
      low = np.array([-3.0, -self.max_steering], dtype=np.float32),
      high = np.array([3.0, self.max_steering], dtype=np.float32),
      dtype = np.float32)
    
    high = np.array(
      [self.area_size, self.area_size, self.max_speed, self.max_speed, np.pi, np.inf, self.area_size, self.area_size, np.pi],
      dtype=np.float32)
    self.observation_space = spaces.Box(low = -high, high = high, dtype=np.float32)

    self.state = None
    self.steps = 0
    self.render_mode = render_mode
    self.viewer = None
    self.seed(seed)
    
  def seed(self, seed: Optional[int] = None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    return [seed]
  
  def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
    if seed is not None:
      self.seed(seed)

    angle = self.np_random.uniform(-np.pi, np.pi)
    dist = self.np_random.uniform(3.0, 6.0)
    x = self.goal_pos[0] + dist * math.cos(angle)
    y = self.goal_pos[1] + dist * math.sin(angle)

    speed = 0.0
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    theta = angle + self.np_random.uniform(-5.0, 5.0)
    omega = 0.0

    self.state = np.array([x, y, vx, vy, theta, omega], dtype=np.float32)
    self.steps = 0

    obs = self._get_obs()
    info = {}
    return obs, info
  
  def _get_obs(self):
    x, y, vx, vy, theta, omega = self.state
    dx = self.goal_pos[0] - x
    dy = self.goal_pos[1] - y
    theta_error = self._angle_normalize(self.goal_theta - theta)
    obs = np.array([x, y, vx, vy, theta, omega, dx, dy, theta_error], dtype=np.float32)
    return obs
  
  @staticmethod
  def _angle_normalize(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi
  
  def step(self, action: np.ndarray):
    assert self.action_space.contains(action), f"Acción inválida: {action}"
    acc = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
    steer = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))

    x, y, vx, vy, theta, omega = self.state
    speed = math.hypot(vx, vy)
    speed = np.clip(speed + acc * self.dt, -self.max_speed, self.max_speed)

    if abs(self.wheelbase) < 1e-6 or abs(math.cos(steer)) < 1e-6:
      theta_dot = 0.0
    else:
      theta_dot = speed / self.wheelbase * math.tan(steer)
    
    theta += theta_dot * self.dt

    x += speed * math.cos(theta) * self.dt
    y += speed * math.sin(theta) * self.dt
    vx = speed * math.cos(theta)
    vy = speed * math.sin(theta)
    omega = theta_dot

    self.state = np.array([x, y, vx, vy, theta, omega], dtype=np.float32)
    obs = self._get_obs()

    dist = math.hypot(obs[6], obs[7])
    ang_err = abs(obs[8])

    reward = -dist
    reward -= ang_err * 0.5

    if dist < 1:
      reward - 0.5 * abs(speed)
    
    done = False
    success = False
    
    if abs(x) > self.area_size or abs(y) > self.area_size:
      reward -= 50
      done = True
    
    if dist < 0.3 and ang_err < math.radians(10) and abs(speed) < 0.5:
      reward += 200
      done = True
      success = True
    
    self.steps += 1
    truncated = False
    if self.steps >= self.max_steps:
      truncated = True
      done = True
    
    info = {"distancia": dist, "error_angular": ang_err, "logrado": success}

    return obs, float(reward), done, truncated, info
  
  def render(self):
    try:
      import matplotlib.pyplot as plt
    except ImportError:
      return None
    
    x, y, vx, vy, theta, omega = self.state
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-self.area_size, self.area_size)
    ax.set_ylim(-self.area_size, self.area_size)
    ax.set_aspect('equal')

    truck_length = 0.6
    truck_width = 0.35
    corners = np.array([
      [truck_length / 2, 0],
      [-truck_length / 2, truck_width / 2],
      [-truck_length / 2, -truck_width / 2],
    ])
    ROTATION = np.array([
      [math.cos(theta), -math.sin(theta)],
      [math.sin(theta), math.cos(theta)]
    ])
    corners_rotate = (ROTATION @ corners.T).T + np.array([x, y])
    ax.fill(corners_rotate[:,0], corners_rotate[:,1], alpha=0.7)

    gx, gy = self.goal_pos
    ax.plot(gx, gy, 'go', markersize=9, label='Espacio')

    ax.set_title(f"Step: {self.steps}")
    plt.show()

  def close(self):
    pass

if __name__ == "__main__":
  env = ParkingEnv(render_mode="human")
  obs, info = env.reset(seed=42)
  print("Obs inicial:", obs)

  total_reward = 0
  for t in range(150):
    dx = obs[6]
    dy = obs[7]
    dist = math.hypot(dx, dy)
    desired_theta = math.atan2(dy, dx)
    theta = obs[4]
    theta_err = ParkingEnv._angle_normalize(desired_theta - theta)

    steer_cmd = np.clip(theta_err, -math.radians(20), math.radians(20))
    accel_cmd = 1 if dist > 1.5 else -0.5 if dist < 0.5 else 0.0

    action = np.array([accel_cmd, steer_cmd], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if t % 20 == 0:
      print(f"t={t}, distancia={info['distancia']:.2f}, error_angular={math.degrees(info['error_angular']):.2f}, reward={reward:.2f}")
    if done:
      print(f"Episodio terminado en {t}")
      break
  print("Recompensa total:", total_reward)
  