"""
Reinforcement Learning Layer Optimizer for Layerwise Adapter

This module implements a reinforcement learning approach to automatically
optimize layer selection strategies for the Layerwise Adapter, balancing
model performance and computational efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import random
from collections import deque, namedtuple
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for reinforcement learning optimization"""
    state_dim: int = 16
    action_dim: int = 10  # Number of possible layer selections
    hidden_dim: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000
    memory_size: int = 10000
    batch_size: int = 32
    target_update: int = 100
    max_episodes: int = 1000


@dataclass
class LayerState:
    """State representation for layer selection environment"""
    layer_importance_scores: List[float]
    model_performance: float
    computational_cost: float
    memory_usage: float
    dataset_sparsity: float
    task_complexity: float
    current_selection: List[bool]  # Which layers are currently selected


@dataclass
class LayerAction:
    """Action representation for layer selection"""
    layer_toggles: List[bool]  # Which layers to toggle on/off
    compression_ratio: float
    distillation_temperature: float


# Named tuple for experience replay
Experience = namedtuple('Experience', 
                       ('state', 'action', 'reward', 'next_state', 'done'))


class PolicyNetwork(nn.Module):
    """Deep Q-Network for layer selection policy"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RewardFunction:
    """Reward function for layer selection optimization"""
    
    def __init__(self, performance_weight: float = 0.6, 
                 efficiency_weight: float = 0.3,
                 memory_weight: float = 0.1):
        self.performance_weight = performance_weight
        self.efficiency_weight = efficiency_weight
        self.memory_weight = memory_weight
        
        # Baseline values for normalization
        self.baseline_performance = 1.0
        self.baseline_efficiency = 1.0
        self.baseline_memory = 1.0
    
    def compute_reward(self, current_state: LayerState, 
                      previous_state: LayerState,
                      action: LayerAction) -> float:
        """Compute reward based on performance and efficiency changes"""
        
        # Performance improvement reward
        perf_change = current_state.model_performance - previous_state.model_performance
        perf_reward = (perf_change / self.baseline_performance) * self.performance_weight
        
        # Efficiency improvement reward (lower cost is better)
        cost_change = previous_state.computational_cost - current_state.computational_cost
        efficiency_reward = (cost_change / self.baseline_efficiency) * self.efficiency_weight
        
        # Memory efficiency reward (lower usage is better)
        memory_change = previous_state.memory_usage - current_state.memory_usage
        memory_reward = (memory_change / self.baseline_memory) * self.memory_weight
        
        # Penalty for too aggressive compression
        num_selected = sum(current_state.current_selection)
        total_layers = len(current_state.current_selection)
        selection_ratio = num_selected / total_layers if total_layers > 0 else 0
        
        # Penalty if selection ratio is too low (< 0.3) or too high (> 0.9)
        selection_penalty = 0
        if selection_ratio < 0.3:
            selection_penalty = -0.5 * (0.3 - selection_ratio)
        elif selection_ratio > 0.9:
            selection_penalty = -0.3 * (selection_ratio - 0.9)
        
        total_reward = perf_reward + efficiency_reward + memory_reward + selection_penalty
        
        return total_reward
    
    def update_baselines(self, performance: float, efficiency: float, memory: float):
        """Update baseline values for reward normalization"""
        self.baseline_performance = max(self.baseline_performance, performance)
        self.baseline_efficiency = max(self.baseline_efficiency, efficiency)
        self.baseline_memory = max(self.baseline_memory, memory)


class LayerSelectionEnvironment:
    """Environment for reinforcement learning layer selection"""
    
    def __init__(self, model: nn.Module, dataloader, 
                 device: torch.device = None):
        self.model = model
        self.dataloader = dataloader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Environment state
        self.current_state = None
        self.episode_steps = 0
        self.max_steps = 50
        
        # Layer information
        self.layer_names = [name for name, _ in model.named_parameters() 
                           if 'weight' in name or 'bias' in name]
        self.num_layers = len(self.layer_names)
        
        # Performance tracking
        self.performance_history = []
        self.best_performance = float('-inf')
        self.best_selection = None
        
        logger.info(f"Initialized LayerSelectionEnvironment with {self.num_layers} layers")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        
        # Initialize with all layers selected
        initial_selection = [True] * self.num_layers
        
        # Compute initial state
        self.current_state = self._compute_state(initial_selection)
        self.episode_steps = 0
        
        return self._state_to_vector(self.current_state)
    
    def step(self, action_vector: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        
        # Decode action
        action = self._decode_action(action_vector)
        
        # Apply action to get new layer selection
        new_selection = self._apply_action(self.current_state.current_selection, action)
        
        # Compute new state
        previous_state = copy.deepcopy(self.current_state)
        self.current_state = self._compute_state(new_selection)
        
        # Compute reward
        reward_func = RewardFunction()
        reward = reward_func.compute_reward(self.current_state, previous_state, action)
        
        # Check if episode is done
        self.episode_steps += 1
        done = (self.episode_steps >= self.max_steps) or self._is_terminal_state()
        
        # Update best performance
        if self.current_state.model_performance > self.best_performance:
            self.best_performance = self.current_state.model_performance
            self.best_selection = new_selection.copy()
        
        # Prepare info
        info = {
            'performance': self.current_state.model_performance,
            'computational_cost': self.current_state.computational_cost,
            'memory_usage': self.current_state.memory_usage,
            'selection_ratio': sum(new_selection) / len(new_selection),
            'episode_steps': self.episode_steps
        }
        
        return self._state_to_vector(self.current_state), reward, done, info
    
    def _compute_state(self, layer_selection: List[bool]) -> LayerState:
        """Compute environment state for given layer selection"""
        
        # Create a temporary model with selected layers
        temp_model = self._create_masked_model(layer_selection)
        
        # Evaluate performance
        performance = self._evaluate_model(temp_model)
        
        # Estimate computational cost
        comp_cost = self._estimate_computational_cost(layer_selection)
        
        # Estimate memory usage
        memory_usage = self._estimate_memory_usage(layer_selection)
        
        # Compute layer importance scores (simplified)
        importance_scores = self._compute_layer_importance(temp_model)
        
        # Dataset characteristics (static for now)
        dataset_sparsity = 0.8  # Placeholder
        task_complexity = 0.6   # Placeholder
        
        return LayerState(
            layer_importance_scores=importance_scores,
            model_performance=performance,
            computational_cost=comp_cost,
            memory_usage=memory_usage,
            dataset_sparsity=dataset_sparsity,
            task_complexity=task_complexity,
            current_selection=layer_selection.copy()
        )
    
    def _create_masked_model(self, layer_selection: List[bool]) -> nn.Module:
        """Create model with selected layers active"""
        
        # Simple approach: zero out weights of unselected layers
        temp_model = copy.deepcopy(self.model)
        
        for i, (name, param) in enumerate(temp_model.named_parameters()):
            if i < len(layer_selection) and not layer_selection[i]:
                # Zero out unselected layer parameters
                param.data.fill_(0.0)
                param.requires_grad = False
        
        return temp_model
    
    def _evaluate_model(self, model: nn.Module, max_batches: int = 10) -> float:
        """Evaluate model performance on a subset of data"""
        
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    # Move batch to device
                    if isinstance(batch, (list, tuple)):
                        batch = [item.to(self.device) if torch.is_tensor(item) else item 
                                for item in batch]
                        
                        if len(batch) >= 2:
                            inputs, targets = batch[0], batch[1]
                        else:
                            inputs, targets = batch[0], None
                    else:
                        inputs = batch.to(self.device)
                        targets = None
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Compute loss
                    if targets is not None:
                        if outputs.shape[-1] == 1:
                            loss = F.mse_loss(outputs.squeeze(), targets.float())
                        else:
                            loss = F.cross_entropy(outputs, targets.long())
                    else:
                        # Use output variance as loss for unsupervised case
                        loss = torch.var(outputs)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
        
        # Return negative loss as performance (higher is better)
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        performance = -avg_loss  # Negative because lower loss = higher performance
        
        return performance
    
    def _estimate_computational_cost(self, layer_selection: List[bool]) -> float:
        """Estimate computational cost based on layer selection"""
        
        # Simple estimation: cost proportional to number of active parameters
        total_params = 0
        active_params = 0
        
        for i, (name, param) in enumerate(self.model.named_parameters()):
            param_count = param.numel()
            total_params += param_count
            
            if i < len(layer_selection) and layer_selection[i]:
                active_params += param_count
        
        # Normalized cost (0 to 1)
        cost = active_params / total_params if total_params > 0 else 1.0
        
        return cost
    
    def _estimate_memory_usage(self, layer_selection: List[bool]) -> float:
        """Estimate memory usage based on layer selection"""
        
        # Similar to computational cost but with different scaling
        selected_ratio = sum(layer_selection) / len(layer_selection) if layer_selection else 1.0
        
        # Memory usage includes activations and gradients
        memory_factor = 1.5  # Approximate factor for activations
        memory_usage = selected_ratio * memory_factor
        
        return memory_usage
    
    def _compute_layer_importance(self, model: nn.Module) -> List[float]:
        """Compute simplified layer importance scores"""
        
        importance_scores = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                # Simple importance: L2 norm of parameters
                importance = torch.norm(param.data).item()
                importance_scores.append(importance)
            else:
                importance_scores.append(0.0)
        
        # Normalize scores
        max_importance = max(importance_scores) if importance_scores else 1.0
        if max_importance > 0:
            importance_scores = [score / max_importance for score in importance_scores]
        
        return importance_scores
    
    def _apply_action(self, current_selection: List[bool], 
                     action: LayerAction) -> List[bool]:
        """Apply action to current layer selection"""
        
        new_selection = current_selection.copy()
        
        # Apply layer toggles
        for i, toggle in enumerate(action.layer_toggles):
            if i < len(new_selection) and toggle:
                new_selection[i] = not new_selection[i]
        
        return new_selection
    
    def _decode_action(self, action_vector: np.ndarray) -> LayerAction:
        """Decode action vector into LayerAction"""
        
        # Simple decoding: first N elements are layer toggles
        layer_toggles = (action_vector[:self.num_layers] > 0.5).tolist() if len(action_vector) >= self.num_layers else [False] * self.num_layers
        
        # Additional action parameters
        compression_ratio = 0.5  # Fixed for now
        distillation_temperature = 3.0  # Fixed for now
        
        return LayerAction(
            layer_toggles=layer_toggles,
            compression_ratio=compression_ratio,
            distillation_temperature=distillation_temperature
        )
    
    def _state_to_vector(self, state: LayerState) -> np.ndarray:
        """Convert LayerState to vector representation"""
        
        vector = []
        
        # Layer importance scores (truncated/padded to fixed size)
        importance_vec = state.layer_importance_scores[:10] if len(state.layer_importance_scores) >= 10 else state.layer_importance_scores + [0.0] * (10 - len(state.layer_importance_scores))
        vector.extend(importance_vec)
        
        # Scalar features
        vector.extend([
            state.model_performance,
            state.computational_cost,
            state.memory_usage,
            state.dataset_sparsity,
            state.task_complexity,
            sum(state.current_selection) / len(state.current_selection)  # Selection ratio
        ])
        
        return np.array(vector, dtype=np.float32)
    
    def _is_terminal_state(self) -> bool:
        """Check if current state is terminal"""
        
        # Terminal if performance is very poor or all layers are deselected
        if self.current_state.model_performance < -10.0:
            return True
        
        if sum(self.current_state.current_selection) == 0:
            return True
        
        return False


class RLLayerOptimizer:
    """Main class for reinforcement learning-based layer optimization"""
    
    def __init__(self, model: nn.Module, dataloader, 
                 config: RLConfig = None, device: torch.device = None):
        
        self.config = config if config else RLConfig()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create environment
        self.env = LayerSelectionEnvironment(model, dataloader, self.device)
        
        # Neural networks
        self.policy_net = PolicyNetwork(
            self.config.state_dim, 
            self.config.action_dim, 
            self.config.hidden_dim
        ).to(self.device)
        
        self.target_net = PolicyNetwork(
            self.config.state_dim, 
            self.config.action_dim, 
            self.config.hidden_dim
        ).to(self.device)
        
        # Copy policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), 
                                         lr=self.config.learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=self.config.memory_size)
        
        # Training tracking
        self.episode_rewards = []
        self.episode_performances = []
        self.steps_done = 0
        
        # Best found configuration
        self.best_layer_selection = None
        self.best_performance = float('-inf')
        
        logger.info("Initialized RLLayerOptimizer")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        
        # Epsilon for exploration
        epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                 np.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        if training and random.random() < epsilon:
            # Random action
            action = np.random.choice([0, 1], size=self.config.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = (q_values > 0).cpu().numpy().flatten()
        
        if training:
            self.steps_done += 1
        
        return action.astype(float)
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def optimize_model(self):
        """Perform one step of optimization on the policy network"""
        
        if len(self.memory) < self.config.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.config.batch_size)
        batch_exp = Experience(*zip(*batch))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch_exp.state).to(self.device)
        action_batch = torch.FloatTensor(batch_exp.action).to(self.device)
        reward_batch = torch.FloatTensor(batch_exp.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch_exp.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch_exp.done).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        current_q_values = self.policy_net(state_batch)
        
        # Convert actions to indices for gathering
        action_indices = action_batch.argmax(dim=1).unsqueeze(1)
        state_action_values = current_q_values.gather(1, action_indices)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.config.batch_size).to(self.device)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            next_state_values[~done_batch] = next_q_values[~done_batch].max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.config.gamma * next_state_values)
        
        # Compute loss
        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
    
    def train(self, num_episodes: int = None) -> Dict[str, Any]:
        """Train the RL agent to optimize layer selection"""
        
        if num_episodes is None:
            num_episodes = self.config.max_episodes
        
        logger.info(f"Starting RL training for {num_episodes} episodes")
        
        training_history = {
            'episode_rewards': [],
            'episode_performances': [],
            'best_selections': [],
            'optimization_progress': []
        }
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0.0
            episode_performance = []
            
            while True:
                # Select action
                action = self.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Update metrics
                episode_reward += reward
                episode_performance.append(info['performance'])
                
                # Move to next state
                state = next_state
                
                # Optimize model
                self.optimize_model()
                
                if done:
                    break
            
            # Update target network
            if episode % self.config.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Record episode results
            avg_performance = np.mean(episode_performance) if episode_performance else 0.0
            training_history['episode_rewards'].append(episode_reward)
            training_history['episode_performances'].append(avg_performance)
            
            # Update best configuration
            if avg_performance > self.best_performance:
                self.best_performance = avg_performance
                self.best_layer_selection = self.env.best_selection
            
            # Logging
            if episode % 50 == 0:
                logger.info(f"Episode {episode}: Reward={episode_reward:.3f}, "
                           f"Performance={avg_performance:.3f}, "
                           f"Best Performance={self.best_performance:.3f}")
                
                training_history['optimization_progress'].append({
                    'episode': episode,
                    'reward': episode_reward,
                    'performance': avg_performance,
                    'best_performance': self.best_performance,
                    'epsilon': self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * 
                              np.exp(-1. * self.steps_done / self.config.epsilon_decay)
                })
        
        logger.info(f"Training completed. Best performance: {self.best_performance:.3f}")
        
        return training_history
    
    def get_optimal_layer_selection(self) -> Tuple[List[bool], float]:
        """Get the optimal layer selection found during training"""
        
        if self.best_layer_selection is None:
            logger.warning("No optimal selection found. Run training first.")
            return [], 0.0
        
        return self.best_layer_selection, self.best_performance
    
    def evaluate_selection(self, layer_selection: List[bool]) -> Dict[str, float]:
        """Evaluate a specific layer selection"""
        
        # Create state with given selection
        state = self.env._compute_state(layer_selection)
        
        return {
            'performance': state.model_performance,
            'computational_cost': state.computational_cost,
            'memory_usage': state.memory_usage,
            'selection_ratio': sum(layer_selection) / len(layer_selection),
            'num_selected_layers': sum(layer_selection)
        }
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_layer_selection': self.best_layer_selection,
            'best_performance': self.best_performance,
            'steps_done': self.steps_done
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load trained model"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.best_layer_selection = checkpoint.get('best_layer_selection')
        self.best_performance = checkpoint.get('best_performance', float('-inf'))
        self.steps_done = checkpoint.get('steps_done', 0)
        
        logger.info(f"Model loaded from {filepath}")
    
    def generate_optimization_report(self, training_history: Dict[str, Any]) -> str:
        """Generate comprehensive optimization report"""
        
        report_lines = [
            "# Reinforcement Learning Layer Optimization Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Training Configuration",
            f"- **Episodes**: {len(training_history['episode_rewards'])}",
            f"- **State Dimension**: {self.config.state_dim}",
            f"- **Action Dimension**: {self.config.action_dim}",
            f"- **Learning Rate**: {self.config.learning_rate}",
            f"- **Gamma**: {self.config.gamma}",
            "",
            "## Optimization Results",
            f"- **Best Performance**: {self.best_performance:.6f}",
            f"- **Final Episode Reward**: {training_history['episode_rewards'][-1]:.3f}",
            f"- **Average Episode Reward**: {np.mean(training_history['episode_rewards']):.3f}",
            "",
            "## Layer Selection Analysis"
        ]
        
        if self.best_layer_selection:
            num_selected = sum(self.best_layer_selection)
            total_layers = len(self.best_layer_selection)
            selection_ratio = num_selected / total_layers
            
            report_lines.extend([
                f"- **Selected Layers**: {num_selected}/{total_layers} ({selection_ratio:.1%})",
                f"- **Compression Ratio**: {1 - selection_ratio:.1%}",
                "",
                "### Selected Layer Indices",
                str([i for i, selected in enumerate(self.best_layer_selection) if selected])
            ])
        
        return "\n".join(report_lines)
