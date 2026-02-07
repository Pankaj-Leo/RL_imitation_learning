"""
AggreVaTe-inspired (Cost-Sensitive DAgger)
Cost-sensitive imitation using a learned risk model (Q(s,a) as a cost proxy)
Addresses the key insight: not all imitation errors are equal
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from core.policies import MLPPolicy
from core.value_functions import QNetwork, AdvantageEstimator


class AggreVaTe:
    """
    AggreVaTe-inspired: cost-sensitive imitation on learner-visited states
    Key insight: not all imitation errors are equal (some states/actions are higher risk)
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
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy = MLPPolicy(
            state_dim,
            action_dim,
            hidden_dims,
            stochastic=False
        ).to(device)
        
        # Q-network for advantage computation
        self.q_network = QNetwork(
            state_dim,
            action_dim,
            hidden_dims
        ).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Aggregated dataset with advantages
        self.states_buffer = []
        self.actions_buffer = []
        self.advantages_buffer = []
        
    def add_data(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray
    ):
        """Add data with advantage labels"""
        self.states_buffer.append(states)
        self.actions_buffer.append(actions)
        self.advantages_buffer.append(advantages)
        
    def train_q_network(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 64
    ):
        """
        Train risk model Q(s,a) to predict a cost proxy ("advantage" label)
        This learns which (s,a) pairs are higher risk in this toy environment
        """
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(advantages)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for batch_states, batch_actions, batch_adv in dataloader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_adv = batch_adv.to(self.device)
                
                # Predict Q-values
                q_pred = self.q_network(batch_states, batch_actions)
                
                # MSE loss on advantages
                loss = nn.MSELoss()(q_pred, batch_adv)
                
                self.q_optimizer.zero_grad()
                loss.backward()
                self.q_optimizer.step()
                
                total_loss += loss.item()
                
    def train_policy_with_advantages(
        self,
        n_epochs: int = 50,
        batch_size: int = 64,
        n_action_samples: int = 10
    ):
        """
        Cost-sensitive classification
        Weight errors by their advantage (cost)
        """
        if len(self.states_buffer) == 0:
            return
            
        states = np.concatenate(self.states_buffer, axis=0)
        expert_actions = np.concatenate(self.actions_buffer, axis=0)
        advantages = np.concatenate(self.advantages_buffer, axis=0)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(expert_actions),
            torch.FloatTensor(advantages)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for batch_states, batch_expert_actions, batch_adv in dataloader:
                batch_states = batch_states.to(self.device)
                batch_expert_actions = batch_expert_actions.to(self.device)
                batch_adv = batch_adv.to(self.device)
                
                # Predict actions
                predicted_actions = self.policy(batch_states)
                
                # Sample alternative actions
                action_samples = torch.randn(
                    batch_states.shape[0],
                    n_action_samples,
                    self.action_dim
                ).to(self.device)
                action_samples = torch.tanh(action_samples)
                
                # Compute advantages of all actions
                batch_states_expanded = batch_states.unsqueeze(1).expand(
                    -1, n_action_samples, -1
                ).reshape(-1, self.state_dim)
                action_samples_flat = action_samples.reshape(-1, self.action_dim)
                
                with torch.no_grad():
                    q_samples = self.q_network(
                        batch_states_expanded,
                        action_samples_flat
                    ).reshape(batch_states.shape[0], n_action_samples)
                    
                    q_expert = self.q_network(batch_states, batch_expert_actions)
                    advantages_all = q_samples - q_expert.unsqueeze(-1)
                    
                # Cost-sensitive regression loss (per-sample weighting; no batch coupling)
                action_error = torch.mean((predicted_actions - batch_expert_actions) ** 2, dim=-1)  # (B,)

                # Per-sample weights: emphasize high-risk states (as measured by expert-provided proxy)
                # batch_adv is a non-negative risk/penalty proxy for the expert action at each visited state.
                adv = torch.clamp(batch_adv, min=0.0)

                # Normalize within the batch but KEEP weights per-sample (no softmax across samples).
                adv_norm = (adv - adv.min()) / (adv.max() - adv.min() + 1e-8)  # (B,) in [0,1]
                alpha = 5.0  # how much more to weight the highest-risk samples
                weights = 1.0 + alpha * adv_norm  # (B,) in [1, 1+alpha]

                reg_loss = (weights * action_error).mean()

                # Use the learned Q(s,a) (cost proxy) to discourage actions predicted to be worse than expert.
                # 1) Penalize learner action if predicted cost exceeds expert action cost.
                q_pi = self.q_network(batch_states, predicted_actions)          # (B,)
                q_expert = self.q_network(batch_states, batch_expert_actions)   # (B,)
                margin = 0.0
                worse_than_expert = torch.relu(q_pi - q_expert + margin).mean()

                # 2) Optional ranking regularizer: expert should beat random action samples by a margin.
                #    q_samples are predicted costs for sampled actions; we want q_expert + m <= q_samples.
                m_rank = 0.1
                rank_loss = torch.relu(q_expert.unsqueeze(-1) + m_rank - q_samples).mean()

                beta = 0.10   # weight for "avoid worse-than-expert" term
                gamma = 0.05  # weight for ranking regularizer
                total = reg_loss + beta * worse_than_expert + gamma * rank_loss

                self.policy_optimizer.zero_grad()
                total.backward()
                self.policy_optimizer.step()

                total_loss += total.item()
                
    def collect_rollouts(self, env, n_trajectories: int = 10, max_steps: int = 200):
        """Roll out current policy"""
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
        
    def run_aggravate(
        self,
        env,
        expert,
        n_iterations: int = 10,
        n_trajectories_per_iter: int = 10,
        initial_demonstrations: List[Tuple] = None
    ):
        """
        Main AggreVaTe loop
        Similar to DAgger but queries for advantages instead of actions
        """
        print("Running AggreVaTe-inspired (cost-sensitive DAgger)...")
        
        # Initialize with expert demonstrations
        if initial_demonstrations is not None:
            for states, actions in initial_demonstrations:
                # Compute initial advantages (all zeros for expert data)
                advantages = np.zeros(len(states))
                self.add_data(states, actions, advantages)
            self.train_policy_with_advantages()
            
        # AggreVaTe iterations
        for iteration in range(n_iterations):
            print(f"\nIteration {iteration + 1}/{n_iterations}")
            
            # Roll out current policy
            learner_states = self.collect_rollouts(env, n_trajectories_per_iter)
            
            # Query expert for actions AND advantages
            expert_actions = expert.label(learner_states, env)
            expert_advantages = expert.compute_advantages(
                learner_states,
                expert_actions,
                env
            )
            
            # Aggregate
            self.add_data(learner_states, expert_actions, expert_advantages)
            
            # Train risk model Q(s,a) to predict a cost proxy ("advantage" label)
            all_states = np.concatenate(self.states_buffer, axis=0)
            all_actions = np.concatenate(self.actions_buffer, axis=0)
            all_advantages = np.concatenate(self.advantages_buffer, axis=0)
            self.train_q_network(all_states, all_actions, all_advantages)
            
            # Train policy with cost-sensitive classification
            self.train_policy_with_advantages()
            
            # Evaluate
            metrics = self.evaluate(env, n_episodes=5)
            print(f"Success rate: {metrics['success_rate']:.3f}, "
                  f"Mean reward: {metrics['mean_reward']:.2f}")
                  
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action"""
        return self.policy.sample_action(state, deterministic=True)
        
    def evaluate(self, env, n_episodes: int = 10) -> dict:
        """Evaluate policy"""
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
