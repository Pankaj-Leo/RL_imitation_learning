"""
DAgger (Dataset Aggregation)
Interactive imitation learning that achieves O(εT) instead of O(εT²)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from core.policies import MLPPolicy
from tqdm import tqdm


class DAgger:
    """
    DAgger: Dataset Aggregation for imitation learning
    Key insight: Query expert on learner's states, not just expert's states
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
        
        # Aggregated dataset (key difference from BC!)
        self.states_buffer = []
        self.actions_buffer = []
        
    def add_data(self, states: np.ndarray, actions: np.ndarray):
        """
        Add data to aggregated dataset
        This is crucial - we don't discard old data!
        """
        self.states_buffer.append(states)
        self.actions_buffer.append(actions)
        
    def collect_rollouts(
        self,
        env,
        n_trajectories: int = 10,
        max_steps: int = 200
    ) -> np.ndarray:
        """
        Roll out current policy to collect learner states
        These will be labeled by the expert
        """
        all_states = []
        
        for _ in range(n_trajectories):
            state, _ = env.reset()
            done = False
            t = 0
            
            while not done and t < max_steps:
                all_states.append(state)
                action = self.policy.sample_action(state, deterministic=True)
                state, _, done, _, _ = env.step(action)
                t += 1
                
        return np.array(all_states)
        
    def train_iteration(
        self,
        n_epochs: int = 50,
        batch_size: int = 64
    ):
        """
        Train on aggregated dataset
        This is Follow-The-Leader (FTL) with squared loss
        """
        if len(self.states_buffer) == 0:
            return
            
        # Concatenate all aggregated data
        states = np.concatenate(self.states_buffer, axis=0)
        actions = np.concatenate(self.actions_buffer, axis=0)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(actions)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for batch_states, batch_actions in dataloader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                predicted_actions = self.policy(batch_states)
                loss = self.loss_fn(predicted_actions, batch_actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
    def run_dagger(
        self,
        env,
        expert,
        n_iterations: int = 10,
        n_trajectories_per_iter: int = 10,
        initial_demonstrations: List[Tuple] = None
    ):
        """
        Main DAgger loop
        
        1. Initialize with expert demonstrations (optional)
        2. For each iteration:
           a. Roll out current policy
           b. Query expert on visited states
           c. Aggregate data
           d. Retrain policy
        """
        print("Running DAgger...")
        
        # Initialize with expert demonstrations
        if initial_demonstrations is not None:
            for states, actions in initial_demonstrations:
                self.add_data(states, actions)
            print(f"Initialized with {len(self.states_buffer[0])} expert demos")
            self.train_iteration()
            
        # DAgger iterations
        for iteration in range(n_iterations):
            print(f"\nIteration {iteration + 1}/{n_iterations}")
            
            # Roll out current policy
            learner_states = self.collect_rollouts(env, n_trajectories_per_iter)
            print(f"Collected {len(learner_states)} learner states")
            
            # Query expert on these states
            expert_labels = expert.label(learner_states, env)
            
            # Aggregate data (crucial!)
            self.add_data(learner_states, expert_labels)
            
            # Retrain on aggregated dataset
            self.train_iteration()
            
            # Evaluate
            metrics = self.evaluate(env, n_episodes=5)
            print(f"Crash rate: {metrics['crash_rate']:.3f}, "
                  f"Mean reward: {metrics['mean_reward']:.2f}")
                  
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action for state"""
        return self.policy.sample_action(state, deterministic=True)
        
    def evaluate(self, env, n_episodes: int = 10) -> dict:
        """Evaluate policy performance"""
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
        
    def get_dataset_size(self) -> int:
        """Return size of aggregated dataset"""
        if len(self.states_buffer) == 0:
            return 0
        return sum(len(s) for s in self.states_buffer)
        
    def save(self, path: str):
        """Save trained policy and dataset"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'states_buffer': self.states_buffer,
            'actions_buffer': self.actions_buffer
        }, path)
        
    def load(self, path: str):
        """Load trained policy and dataset"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.states_buffer = checkpoint['states_buffer']
        self.actions_buffer = checkpoint['actions_buffer']


class AdaptiveDAgger(DAgger):
    """
    DAgger with adaptive sampling
    Focuses queries on high-uncertainty states
    """
    
    def __init__(self, *args, uncertainty_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_threshold = uncertainty_threshold
        
        # Use ensemble for uncertainty
        from core.policies import EnsemblePolicy
        self.policy = EnsemblePolicy(
            kwargs.get('state_dim'),
            kwargs.get('action_dim'),
            num_models=5
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=kwargs.get('learning_rate', 3e-4))
        
    def collect_rollouts(self, env, n_trajectories: int = 10, max_steps: int = 200):
        """Collect rollouts and track uncertainty"""
        all_states = []
        all_uncertainties = []
        
        for _ in range(n_trajectories):
            state, _ = env.reset()
            done = False
            t = 0
            
            while not done and t < max_steps:
                action, uncertainty = self.policy.sample_action(
                    state, return_uncertainty=True
                )
                all_states.append(state)
                all_uncertainties.append(uncertainty)
                
                state, _, done, _, _ = env.step(action)
                t += 1
                
        # Prioritize high-uncertainty states
        states = np.array(all_states)
        uncertainties = np.array(all_uncertainties)
        
        # Only query expert on uncertain states
        uncertain_mask = uncertainties > self.uncertainty_threshold
        if uncertain_mask.sum() > 0:
            return states[uncertain_mask]
        else:
            # Fall back to random sampling
            n_samples = min(100, len(states))
            indices = np.random.choice(len(states), n_samples, replace=False)
            return states[indices]
