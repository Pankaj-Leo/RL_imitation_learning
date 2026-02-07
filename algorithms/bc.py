"""
Behavior Cloning (BC)
The broken baseline that demonstrates O(εT²) error compounding
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from core.policies import MLPPolicy
from tqdm import tqdm


class BehaviorCloning:
    """
    Standard supervised learning approach to imitation
    Trains on expert states, fails catastrophically at test time
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        device: str = 'cpu'
    ):
        self.device = device
        self.policy = MLPPolicy(
            state_dim, 
            action_dim, 
            hidden_dims,
            stochastic=False
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def train(
        self,
        demonstrations: List[Tuple[np.ndarray, np.ndarray]],
        n_epochs: int = 100,
        batch_size: int = 64
    ):
        """
        Train policy on expert demonstrations
        demonstrations: List of (states, actions) tuples
        """
        # Aggregate all demonstrations
        all_states = []
        all_actions = []
        
        for states, actions in demonstrations:
            all_states.append(states)
            all_actions.append(actions)
            
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(actions)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        print("Training Behavior Cloning...")
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for batch_states, batch_actions in dataloader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                # Forward pass
                predicted_actions = self.policy(batch_states)
                loss = self.loss_fn(predicted_actions, batch_actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
                
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action for a given state"""
        return self.policy.sample_action(state, deterministic=True)
        
    def evaluate(self, env, n_episodes: int = 10) -> dict:
        """
        Evaluate policy performance
        Returns metrics showing catastrophic failure
        """
        rewards = []
        fell_off_cliff = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            t = 0
            
            while not done and t < 200:
                action = self.predict(state)
                state, reward, done, _, info = env.step(action)
                episode_reward += reward
                t += 1
                
            rewards.append(episode_reward)
            fell_off_cliff.append(info.get('fell_off_cliff', False))
            episode_lengths.append(t)
            
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'crash_rate': np.mean(fell_off_cliff),
            'mean_episode_length': np.mean(episode_lengths),
            'success_rate': 1.0 - np.mean(fell_off_cliff)
        }
        
    def compute_training_error(self, demonstrations: List[Tuple]) -> float:
        """
        Compute training error (misleadingly low!)
        Shows that low training error doesn't guarantee good performance
        """
        all_states = []
        all_actions = []
        
        for states, actions in demonstrations:
            all_states.append(states)
            all_actions.append(actions)
            
        states = torch.FloatTensor(np.concatenate(all_states, axis=0))
        actions = torch.FloatTensor(np.concatenate(all_actions, axis=0))
        
        with torch.no_grad():
            predicted = self.policy(states)
            error = torch.mean((predicted - actions) ** 2).item()
            
        return error
        
    def save(self, path: str):
        """Save trained policy"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """Load trained policy"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class BehaviorCloningWithFeedback(BehaviorCloning):
    """
    BC with previous action as feature
    Demonstrates dangerous feedback loops
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        include_prev_action: bool = True,
        **kwargs
    ):
        # Don't call super().__init__ yet
        self.device = kwargs.get('device', 'cpu')
        
        from core.policies import FeedbackAwarePolicy
        self.policy = FeedbackAwarePolicy(
            state_dim,
            action_dim,
            include_prev_action=include_prev_action,
            hidden_dims=kwargs.get('hidden_dims', [256, 256]),
            stochastic=False
        ).to(self.device)
        
        learning_rate = kwargs.get('learning_rate', 3e-4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def predict(self, state: np.ndarray, reset: bool = False) -> np.ndarray:
        """Predict with feedback loop tracking"""
        if reset:
            self.policy.reset_history()
        return self.policy.sample_action(state, deterministic=True)
        
    def evaluate(self, env, n_episodes: int = 10, verbose: bool = False) -> dict:
        """Evaluate with feedback loop detection"""
        rewards = []
        fell_off_cliff = []
        episode_lengths = []
        stuck_in_loop = []
        
        for ep in range(n_episodes):
            state, _ = env.reset()
            self.policy.reset_history()
            
            episode_reward = 0
            done = False
            t = 0
            action_history = []
            
            while not done and t < 200:
                action = self.predict(state)
                action_history.append(action)
                
                state, reward, done, _, info = env.step(action)
                episode_reward += reward
                t += 1
                
                # Detect feedback loop (repeating actions)
                if len(action_history) > 10:
                    recent = np.array(action_history[-10:])
                    if np.std(recent) < 0.01:  # Stuck!
                        stuck_in_loop.append(True)
                        if verbose:
                            print(f"Episode {ep}: Stuck in feedback loop at t={t}")
                        break
                        
            if t >= 200 or not done:
                stuck_in_loop.append(False)
                
            rewards.append(episode_reward)
            fell_off_cliff.append(info.get('fell_off_cliff', False))
            episode_lengths.append(t)
            
        return {
            'mean_reward': np.mean(rewards),
            'crash_rate': np.mean(fell_off_cliff),
            'feedback_loop_rate': np.mean(stuck_in_loop),
            'mean_episode_length': np.mean(episode_lengths)
        }
