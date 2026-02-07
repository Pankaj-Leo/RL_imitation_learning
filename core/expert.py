"""
Expert Demonstrators for Imitation Learning
Provides optimal and sub-optimal experts for various tasks
"""
import numpy as np
from typing import List, Tuple, Optional
import torch


class OptimalRacecarDriver:
    """
    Expert driver for the CliffRacetrack environment
    Uses simple PID control to follow the centerline
    """
    
    def __init__(self, 
                 lookahead: float = 2.0,
                 kp: float = 0.8,
                 kd: float = 0.3):
        self.lookahead = lookahead
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0
        
    def get_action(self, state: np.ndarray, env) -> np.ndarray:
        """
        Compute optimal action using PID controller
        state: [x, y, velocity, heading]
        """
        x, y, velocity, heading = state
        
        # Get centerline at current position
        centerline = env.get_track_centerline()
        
        # Find closest point on centerline
        distances = np.linalg.norm(centerline - np.array([x, y]), axis=1)
        closest_idx = np.argmin(distances)
        
        # Look ahead on centerline
        lookahead_idx = min(closest_idx + int(self.lookahead / 0.5), len(centerline) - 1)
        target_point = centerline[lookahead_idx]
        
        # Compute lateral error
        lateral_error = y - target_point[1]
        
        # PID control for steering
        d_error = lateral_error - self.prev_error
        steering = -self.kp * lateral_error - self.kd * d_error
        steering = np.clip(steering, -1.0, 1.0)
        
        # Throttle control - slow down if off centerline
        if abs(lateral_error) > 0.5:
            throttle = -0.3  # Brake
        else:
            throttle = 0.5  # Accelerate
            
        self.prev_error = lateral_error
        
        return np.array([steering, throttle])
        
    def demonstrate(self, env, n_trajectories: int = 50) -> List[Tuple]:
        """
        Collect expert demonstrations
        Returns: List of (states, actions) tuples
        """
        demonstrations = []
        
        for _ in range(n_trajectories):
            states = []
            actions = []
            
            state, _ = env.reset()
            done = False
            self.prev_error = 0.0
            
            while not done:
                action = self.get_action(state, env)
                states.append(state)
                actions.append(action)
                
                state, _, done, _, _ = env.step(action)
                
            demonstrations.append((
                np.array(states),
                np.array(actions)
            ))
            
        return demonstrations
        
    def label(self, states: np.ndarray, env) -> np.ndarray:
        """
        Label arbitrary states with expert actions
        Used by DAgger for interactive queries
        """
        actions = []
        for state in states:
            action = self.get_action(state, env)
            actions.append(action)
        return np.array(actions)
        
    def compute_advantages(self, states: np.ndarray, actions: np.ndarray, env) -> np.ndarray:
        """
        Compute advantage of each action for AggreVaTe
        A(s,a) = cost_penalty if action leads off track, else 0
        """
        advantages = np.zeros(len(states))
        
        for i, (state, action) in enumerate(zip(states, actions)):
            # Simulate one step
            x, y, velocity, heading = state
            steering, throttle = action
            
            # Simple dynamics
            new_heading = heading + steering * 0.1
            new_velocity = velocity + throttle * 0.1
            new_x = x + new_velocity * np.cos(new_heading) * 0.1
            new_y = y + new_velocity * np.sin(new_heading) * 0.1
            
            # High advantage if going off track
            if new_y < -env.track_width:
                advantages[i] = 100.0  # Very bad
            elif abs(new_y) > env.track_width:
                advantages[i] = 10.0 * abs(new_y)  # Bad
            else:
                # Small advantage based on distance from centerline
                centerline = env.get_track_centerline()
                distances = np.linalg.norm(centerline - np.array([new_x, new_y]), axis=1)
                advantages[i] = np.min(distances)
                
        return advantages


class SuboptimalDriver:
    """
    Intentionally suboptimal driver for testing IRL robustness
    Takes unnecessarily long paths but stays safe
    """
    
    def __init__(self, optimality: float = 0.7):
        self.optimality = optimality  # 0-1, lower = more suboptimal
        self.optimal_driver = OptimalRacecarDriver()
        
    def get_action(self, state: np.ndarray, env) -> np.ndarray:
        """Mix optimal action with random noise"""
        optimal_action = self.optimal_driver.get_action(state, env)
        
        # Add suboptimality
        if np.random.rand() > self.optimality:
            # Make conservative mistake
            noise = np.random.randn(2) * 0.3
            noise[1] = abs(noise[1]) - 0.5  # Always brake more
            action = optimal_action + noise
        else:
            action = optimal_action
            
        return np.clip(action, -1.0, 1.0)
        
    def demonstrate(self, env, n_trajectories: int = 50):
        """Collect suboptimal demonstrations"""
        demonstrations = []
        
        for _ in range(n_trajectories):
            states = []
            actions = []
            
            state, _ = env.reset()
            done = False
            
            while not done:
                action = self.get_action(state, env)
                states.append(state)
                actions.append(action)
                
                state, _, done, _, _ = env.step(action)
                
            demonstrations.append((
                np.array(states),
                np.array(actions)
            ))
            
        return demonstrations


class HumanInterventionSimulator:
    """
    Simulates human interventions based on value thresholds
    Used for intervention-based learning experiments
    """
    
    def __init__(self, 
                 intervention_threshold: float = 5.0,
                 return_threshold: float = 1.0):
        self.intervention_threshold = intervention_threshold
        self.return_threshold = return_threshold
        self.optimal_driver = OptimalRacecarDriver()
        self.in_control = False
        
    def should_intervene(self, state: np.ndarray, action: np.ndarray, env) -> bool:
        """Decide if human should take over"""
        # Check if action leads to dangerous state
        x, y, velocity, heading = state
        steering, throttle = action
        
        # Simulate one step
        new_heading = heading + steering * 0.1
        new_y = y + velocity * np.sin(new_heading) * 0.1
        
        # Intervene if approaching cliff or off track
        if abs(new_y) > self.intervention_threshold or new_y < -1.5:
            return True
        return False
        
    def should_return_control(self, state: np.ndarray, env) -> bool:
        """Decide if human should return control to robot"""
        x, y, velocity, heading = state
        
        # Return control if back on track
        if abs(y) < self.return_threshold:
            return True
        return False
        
    def intervene(self, state: np.ndarray, env) -> np.ndarray:
        """Take corrective action"""
        self.in_control = True
        return self.optimal_driver.get_action(state, env)
        
    def simulate_intervention_data(self, env, learner_policy, n_episodes: int = 20):
        """
        Simulate robot-human interaction with interventions
        Returns intervention data with level labels
        """
        intervention_data = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            trajectory = []
            intervention_segments = []
            
            while not done:
                # Robot tries to act
                action = learner_policy.sample_action(state)
                
                # Check if intervention needed
                if self.should_intervene(state, action, env):
                    intervention_start = len(trajectory)
                    self.in_control = True
                    
                    # Human takes over
                    while self.in_control and not done:
                        action = self.intervene(state, env)
                        trajectory.append((state, action, 'level3'))
                        
                        state, _, done, _, _ = env.step(action)
                        
                        if self.should_return_control(state, env):
                            self.in_control = False
                            intervention_end = len(trajectory)
                            intervention_segments.append((intervention_start, intervention_end))
                else:
                    # Robot in control
                    trajectory.append((state, action, 'level2'))
                    state, _, done, _, _ = env.step(action)
                    
            intervention_data.append((trajectory, intervention_segments))
            
        return intervention_data
