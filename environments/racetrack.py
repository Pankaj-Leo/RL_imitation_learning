"""
Cliff Racetrack Environment
Demonstrates catastrophic covariate shift in imitation learning
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class CliffRacetrack(gym.Env):
    """
    A racetrack environment where the track runs along a cliff edge.
    Making mistakes leads to falling off the cliff - perfect for demonstrating
    how BC compounds errors while DAgger doesn't.
    
    State: [x_position, y_position, velocity, heading]
    Action: [steering, throttle]
    """
    
    def __init__(self, track_width: float = 2.0, cliff_penalty: float = 100.0):
        super().__init__()
        
        self.track_width = track_width
        self.cliff_penalty = cliff_penalty
        self.track_length = 50.0
        
        # State: [x, y, velocity, heading]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -5.0, 0.0, -np.pi]),
            high=np.array([self.track_length, 5.0, 10.0, np.pi]),
            dtype=np.float32
        )
        
        # Action: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        self.dt = 0.1
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Start at beginning of track, centered, low velocity
        self.state = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.time = 0
        self.fell_off_cliff = False
        
        return self.state.copy(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        steering, throttle = action
        x, y, velocity, heading = self.state
        
        # Simple bicycle model dynamics
        heading += steering * self.dt
        velocity += throttle * self.dt
        velocity = np.clip(velocity, 0.0, 10.0)
        
        x += velocity * np.cos(heading) * self.dt
        y += velocity * np.sin(heading) * self.dt
        
        # Compute cost
        cost = 0.0
        done = False
        
        # Track boundaries (cliff on one side)
        if y < -self.track_width:
            # Fell off cliff!
            cost = self.cliff_penalty
            done = True
            self.fell_off_cliff = True
        elif y > self.track_width:
            # Went off track (other side)
            cost = 10.0 * abs(y - self.track_width)
        else:
            # On track - small cost for time
            cost = 0.1
            
        # Reached goal
        if x >= self.track_length:
            done = True
            
        self.state = np.array([x, y, velocity, heading], dtype=np.float32)
        self.time += 1
        
        info = {
            'fell_off_cliff': self.fell_off_cliff,
            'off_track': abs(y) > self.track_width,
            'position': (x, y)
        }
        
        return self.state.copy(), -cost, done, False, info
    
    def get_track_centerline(self, num_points: int = 100) -> np.ndarray:
        """Returns points along the ideal racing line"""
        x = np.linspace(0, self.track_length, num_points)
        
        # Add some curves to make it interesting
        y = 0.3 * np.sin(0.2 * x) + 0.2 * np.sin(0.5 * x)
        
        return np.stack([x, y], axis=1)
    
    def render(self, mode='human'):
        """Visualize the current state"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(12, 4))
            
        self.ax.clear()
        
        # Draw track boundaries
        x_track = np.linspace(0, self.track_length, 100)
        upper_bound = np.ones_like(x_track) * self.track_width
        lower_bound = np.ones_like(x_track) * -self.track_width
        
        self.ax.fill_between(x_track, lower_bound, -5, alpha=0.3, color='red', label='Cliff')
        self.ax.fill_between(x_track, lower_bound, upper_bound, alpha=0.2, color='gray', label='Track')
        
        # Draw centerline
        centerline = self.get_track_centerline()
        self.ax.plot(centerline[:, 0], centerline[:, 1], 'k--', alpha=0.5, label='Ideal line')
        
        # Draw vehicle
        x, y, velocity, heading = self.state
        self.ax.plot(x, y, 'bo', markersize=10, label='Vehicle')
        
        # Draw heading
        dx = 0.5 * np.cos(heading)
        dy = 0.5 * np.sin(heading)
        self.ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        self.ax.set_xlim(-1, self.track_length + 1)
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlabel('Track Position (m)')
        self.ax.set_ylabel('Lateral Position (m)')
        self.ax.set_title(f'Cliff Racetrack - Time: {self.time}')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        plt.pause(0.01)
        
        return self.fig


class NavigationWithObstacles(gym.Env):
    """
    Multi-agent navigation environment for testing concepts and game-theoretic planning
    """
    
    def __init__(self, num_agents: int = 5, grid_size: int = 20):
        super().__init__()
        
        self.num_agents = num_agents
        self.grid_size = grid_size
        
        # State: robot position + velocities + other agent positions + velocities
        state_dim = 2 + 2 + (num_agents - 1) * 4
        self.observation_space = spaces.Box(
            low=-grid_size, high=grid_size, shape=(state_dim,), dtype=np.float32
        )
        
        # Action: [vx, vy]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Robot starts at left, goal at right
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_vel = np.array([0.0, 0.0])
        self.goal = np.array([self.grid_size - 2.0, 0.0])
        
        # Other agents with random positions and velocities
        self.agent_positions = np.random.uniform(
            -self.grid_size/2, self.grid_size/2, 
            size=(self.num_agents - 1, 2)
        )
        self.agent_velocities = np.random.uniform(-0.5, 0.5, size=(self.num_agents - 1, 2))
        
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation vector"""
        obs = np.concatenate([
            self.robot_pos,
            self.robot_vel,
            self.agent_positions.flatten(),
            self.agent_velocities.flatten()
        ])
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Update robot
        self.robot_vel = np.clip(action, -1.0, 1.0)
        self.robot_pos += self.robot_vel * 0.1
        
        # Update other agents (simple motion model)
        self.agent_positions += self.agent_velocities * 0.1
        
        # Keep agents in bounds (bounce off walls)
        for i in range(len(self.agent_positions)):
            for dim in [0, 1]:
                if abs(self.agent_positions[i, dim]) > self.grid_size / 2:
                    self.agent_velocities[i, dim] *= -1
                    self.agent_positions[i, dim] = np.clip(
                        self.agent_positions[i, dim],
                        -self.grid_size / 2,
                        self.grid_size / 2
                    )
        
        # Compute reward
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal)
        reward = -dist_to_goal
        
        # Collision penalty
        for agent_pos in self.agent_positions:
            dist = np.linalg.norm(self.robot_pos - agent_pos)
            if dist < 1.0:
                reward -= 10.0
                
        # Goal bonus
        done = dist_to_goal < 0.5
        if done:
            reward += 100.0
            
        info = {'distance_to_goal': dist_to_goal}
        
        return self._get_obs(), reward, done, False, info
