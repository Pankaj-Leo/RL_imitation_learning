"""
Value Functions and Q-Networks for Imitation Learning
Used in AggreVaTe, IRL, and game-theoretic approaches
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class QNetwork(nn.Module):
    """
    Q-function approximator Q(s, a) -> R
    Critical for advantage-based imitation learning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a)"""
        x = torch.cat([state, action], dim=-1)
        return self.network(x).squeeze(-1)
        
    def compute_advantages(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        expert_action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantage of each action relative to expert action
        A(s, a) = Q(s, a) - Q(s, a_expert)
        """
        q_actions = self.forward(state, actions)
        q_expert = self.forward(state, expert_action)
        
        return q_actions - q_expert.unsqueeze(-1)


class ValueNetwork(nn.Module):
    """
    State value function V(s) -> R
    Used in policy gradient and advantage estimation
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [256, 256]
    ):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute V(s)"""
        return self.network(state).squeeze(-1)


class CostNetwork(nn.Module):
    """
    Cost function C(s, a) for IRL
    The "discriminator" in the game-theoretic framework
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 64,
        hidden_dims: list = [256, 256]
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # Feature extractor
        feat_layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims[:-1]:
            feat_layers.append(nn.Linear(input_dim, hidden_dim))
            feat_layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        feat_layers.append(nn.Linear(input_dim, feature_dim))
        self.feature_extractor = nn.Sequential(*feat_layers)
        
        # Cost head
        self.cost_head = nn.Linear(feature_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute cost C(s, a)"""
        x = torch.cat([state, action], dim=-1)
        features = self.feature_extractor(x)
        cost = self.cost_head(features)
        return cost.squeeze(-1)
        
    def get_features(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Extract features for moment matching"""
        x = torch.cat([state, action], dim=-1)
        return self.feature_extractor(x)
        
    def compute_trajectory_cost(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute total cost of a trajectory"""
        costs = self.forward(states, actions)
        return costs.sum(dim=0)


class LinearCostFunction(nn.Module):
    """
    Linear cost function C(s,a) = w^T * features(s,a)
    Used in max-margin IRL and feature matching
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(feature_dim))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute cost from features"""
        return torch.matmul(features, self.weights)
        
    def get_weights(self) -> np.ndarray:
        """Return cost weights as numpy array"""
        return self.weights.detach().cpu().numpy()


class AdvantageEstimator:
    """
    Utility class for computing advantages from Q-functions
    Handles both Monte Carlo and bootstrapped estimates
    """
    
    def __init__(
        self,
        q_network: QNetwork,
        gamma: float = 0.99
    ):
        self.q_network = q_network
        self.gamma = gamma
        
    def compute_advantages_monte_carlo(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        expert_actions: np.ndarray
    ) -> np.ndarray:
        """
        Compute advantages using Monte Carlo returns
        A(s,a) = Q(s,a) - Q(s,a_expert)
        where Q is estimated from actual returns
        """
        T = len(states)
        returns = np.zeros(T)
        
        # Compute returns backward
        returns[-1] = rewards[-1]
        for t in range(T-2, -1, -1):
            returns[t] = rewards[t] + self.gamma * returns[t+1]
            
        # Convert to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        expert_actions_t = torch.FloatTensor(expert_actions)
        
        with torch.no_grad():
            q_actions = self.q_network(states_t, actions_t)
            q_expert = self.q_network(states_t, expert_actions_t)
            advantages = q_actions - q_expert
            
        return advantages.cpu().numpy()
        
    def compute_advantages_td(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        expert_actions: np.ndarray,
        next_expert_actions: np.ndarray
    ) -> np.ndarray:
        """
        Compute advantages using TD learning
        A(s,a) = r + γV(s') - Q(s,a_expert)
        """
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        next_states_t = torch.FloatTensor(next_states)
        expert_actions_t = torch.FloatTensor(expert_actions)
        next_expert_actions_t = torch.FloatTensor(next_expert_actions)
        rewards_t = torch.FloatTensor(rewards)
        
        with torch.no_grad():
            # Q(s', a_expert) as proxy for V(s')
            v_next = self.q_network(next_states_t, next_expert_actions_t)
            q_expert = self.q_network(states_t, expert_actions_t)
            
            # TD target
            td_target = rewards_t + self.gamma * v_next
            advantages = td_target - q_expert
            
        return advantages.cpu().numpy()
