"""
Neural Network Policies for Imitation Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MLPPolicy(nn.Module):
    """
    Multi-layer perceptron policy network
    Maps states to actions (deterministic or stochastic)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = 'relu',
        stochastic: bool = False
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stochastic = stochastic
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim
            
        self.backbone = nn.Sequential(*layers)
        
        if stochastic:
            # Output mean and log_std for Gaussian policy
            self.mean_head = nn.Linear(input_dim, action_dim)
            self.log_std_head = nn.Linear(input_dim, action_dim)
        else:
            # Deterministic policy
            self.action_head = nn.Linear(input_dim, action_dim)
            
    def _get_activation(self, name: str):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {name}")
            
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action or action distribution"""
        features = self.backbone(state)
        
        if self.stochastic:
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            return mean, log_std
        else:
            action = torch.tanh(self.action_head(features))
            return action
            
    def sample_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """Sample action from policy (numpy interface)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            if self.stochastic:
                mean, log_std = self.forward(state_tensor)
                
                if deterministic:
                    action = mean
                else:
                    std = torch.exp(log_std)
                    noise = torch.randn_like(mean)
                    action = mean + std * noise
                    
                action = torch.tanh(action)  # Bound to [-1, 1]
            else:
                action = self.forward(state_tensor)
                
            return action.squeeze(0).cpu().numpy()
            
    def get_log_prob(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of action (for stochastic policies)"""
        if not self.stochastic:
            raise ValueError("get_log_prob only for stochastic policies")
            
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Gaussian log prob
        log_prob = -0.5 * (
            ((action - mean) / std) ** 2 + 
            2 * log_std + 
            np.log(2 * np.pi)
        )
        
        return log_prob.sum(dim=-1)


class FeedbackAwarePolicy(MLPPolicy):
    """
    Policy that explicitly models previous actions
    Used to demonstrate feedback loops and their dangers
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        include_prev_action: bool = True,
        **kwargs
    ):
        self.include_prev_action = include_prev_action
        
        # Augment state with previous action if needed
        if include_prev_action:
            state_dim += action_dim
            
        super().__init__(state_dim, action_dim, **kwargs)
        self.prev_action = None
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional previous action concatenation"""
        if self.include_prev_action and self.prev_action is not None:
            batch_size = state.shape[0]
            prev_action_expanded = self.prev_action.expand(batch_size, -1)
            state = torch.cat([state, prev_action_expanded], dim=-1)
            
        return super().forward(state)
        
    def sample_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample action and store it for next timestep"""
        action = super().sample_action(state, deterministic)
        
        if self.include_prev_action:
            self.prev_action = torch.FloatTensor(action).unsqueeze(0)
            
        return action
        
    def reset_history(self):
        """Reset previous action (call at episode start)"""
        self.prev_action = None


class EnsemblePolicy(nn.Module):
    """
    Ensemble of policies for uncertainty quantification
    Useful for active learning and intervention detection
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_models: int = 5,
        hidden_dims: list = [256, 256]
    ):
        super().__init__()
        
        self.num_models = num_models
        self.policies = nn.ModuleList([
            MLPPolicy(state_dim, action_dim, hidden_dims, stochastic=True)
            for _ in range(num_models)
        ])
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and variance across ensemble"""
        all_means = []
        
        for policy in self.policies:
            mean, _ = policy(state)
            all_means.append(mean)
            
        all_means = torch.stack(all_means)
        ensemble_mean = all_means.mean(dim=0)
        ensemble_var = all_means.var(dim=0)
        
        return ensemble_mean, ensemble_var
        
    def sample_action(
        self, 
        state: np.ndarray, 
        return_uncertainty: bool = False
    ) -> np.ndarray:
        """Sample from ensemble, optionally return uncertainty"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, var = self.forward(state_tensor)
            
            action = mean.squeeze(0).cpu().numpy()
            
            if return_uncertainty:
                uncertainty = var.mean().item()
                return action, uncertainty
            else:
                return action
