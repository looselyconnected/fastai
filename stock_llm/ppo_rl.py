"""
PPO (Proximal Policy Optimization) implementation for stock prediction model.

This module implements PPO training to optimize the model for 20-day cumulative direction prediction.
PPO provides better stability and sample efficiency compared to vanilla policy gradient methods.

Key improvements over vanilla REINFORCE:
- Clipped objective function for stable updates
- Multiple epochs per batch for better sample efficiency
- Separate value function for variance reduction
- Generalized Advantage Estimation (GAE)
- Batch processing for efficiency
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Tuple, Dict, Optional, NamedTuple
import random
import logging
from collections import deque
import math

from model import GPT, load_model, GPTConfig
from data import data_columns, get_data_for_eval, decode_data, encode_data
from stockdata import StockData

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOExperience(NamedTuple):
    """Experience tuple for PPO training."""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor

class PPOBuffer:
    """Experience buffer for PPO training."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.returns = []
        self.advantages = []
    
    def add(self, state, action, reward, value, log_prob, done):
        """Add experience to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        # Compute returns and advantages
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # GAE computation
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get_batch(self, batch_size: int):
        """Get a batch of experiences."""
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_returns = [self.returns[i] for i in indices]
        batch_advantages = [self.advantages[i] for i in indices]
        batch_old_log_probs = [self.log_probs[i] for i in indices]
        batch_values = [self.values[i] for i in indices]
        
        return (batch_states, batch_actions, batch_returns, 
                batch_advantages, batch_old_log_probs, batch_values)
    
    def size(self):
        """Return buffer size."""
        return len(self.states)


class PPOValueNetwork(nn.Module):
    """Separate value network for PPO."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class PPOTrainer:
    """
    PPO trainer for stock prediction model.
    
    Uses Proximal Policy Optimization for more stable and sample-efficient training
    compared to vanilla policy gradient methods.
    """
    
    def __init__(self, 
                 model: GPT,
                 device: str = 'cpu',
                 learning_rate: float = 5e-6,  # Reduced for stability
                 value_lr: float = 5e-5,       # Reduced for stability
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.1,    # More conservative clipping
                 epochs_per_update: int = 3,   # Fewer epochs to prevent overfitting
                 batch_size: int = 32,         # Smaller batches for stability
                 value_loss_coeff: float = 0.25, # Reduced value loss weight
                 entropy_coeff: float = 0.05,  # Higher entropy for exploration
                 max_grad_norm: float = 0.5,
                 buffer_size: int = 1024,      # Smaller buffer for faster updates
                 reward_scale: float = 0.1):   # Scale rewards to prevent extreme values
        """
        Initialize PPO trainer.
        
        Args:
            model: Pre-trained GPT model
            device: Device to run on
            learning_rate: Learning rate for policy
            value_lr: Learning rate for value function
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            epochs_per_update: Number of epochs per update
            batch_size: Batch size for updates
            value_loss_coeff: Value loss coefficient
            entropy_coeff: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            buffer_size: Experience buffer size
        """
        self.model = model
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.reward_scale = reward_scale
        
        # Create value network
        self.value_net = PPOValueNetwork(model.config.n_embd).to(device)
        
        # Create optimizers
        self.policy_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)
        
        # Experience buffer
        self.buffer = PPOBuffer(buffer_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.prediction_accuracy = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        
        # Policy collapse detection
        self.recent_predictions = deque(maxlen=50)  # Track recent predictions
        self.collapse_threshold = 0.05  # If std dev of predictions < this, consider collapsed
        
        print(f"✓ PPO Trainer initialized")
        print(f"  Policy LR: {learning_rate}, Value LR: {value_lr}")
        print(f"  Clip ε: {clip_epsilon}, Epochs: {epochs_per_update}")
        print(f"  Batch size: {batch_size}, Buffer size: {buffer_size}")
        print(f"  Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    def compute_continuous_reward(self, 
                                predicted_tokens: torch.Tensor, 
                                actual_tokens: torch.Tensor,
                                debug: bool = False) -> float:
        """
        Compute continuous reward based on direction matching and magnitude alignment.
        This function is also used for diagnostics in the training loop.
        """
        # Ensure tokens are properly shaped
        if len(predicted_tokens) % len(data_columns) != 0:
            print(f"⚠️  Warning: predicted_tokens length {len(predicted_tokens)} not divisible by {len(data_columns)}")
            return 0.0
        
        if len(actual_tokens) % len(data_columns) != 0:
            print(f"⚠️  Warning: actual_tokens length {len(actual_tokens)} not divisible by {len(data_columns)}")
            return 0.0
        
        # Decode tokens to get close_bucket values
        try:
            pred_df = decode_data(predicted_tokens)
            actual_df = decode_data(actual_tokens)
        except Exception as e:
            print(f"❌ Error decoding tokens: {e}")
            return 0.0
        
        # Convert close_bucket tokens to standardized values
        def token_to_std_value(token):
            try:
                idx = int(token - StockData.CLOSE_LABELS.min())
                if 0 <= idx < len(StockData.BIN_VALUES):
                    return StockData.BIN_VALUES[idx]
                else:
                    return 0.0
            except:
                return 0.0
        
        # Get close values and compute cumulative movement
        if isinstance(pred_df, pd.DataFrame) and 'close_bucket' in pred_df.columns:
            pred_close_tokens = pred_df['close_bucket'].values
            pred_std_values = [token_to_std_value(token) for token in pred_close_tokens]
            pred_cumulative = sum(pred_std_values)
        else:
            pred_cumulative = 0.0
        
        if isinstance(actual_df, pd.DataFrame) and 'close_bucket' in actual_df.columns:
            actual_close_tokens = actual_df['close_bucket'].values
            actual_std_values = [token_to_std_value(token) for token in actual_close_tokens]
            actual_cumulative = sum(actual_std_values)
        else:
            actual_cumulative = 0.0
        
        # Determine directions
        pred_direction = 1 if pred_cumulative > 0 else (-1 if pred_cumulative < 0 else 0)
        actual_direction = 1 if actual_cumulative > 0 else (-1 if actual_cumulative < 0 else 0)
        
        # Compute magnitude difference
        magnitude_diff = abs(pred_cumulative - actual_cumulative)
        log_penalty = min(np.log10(magnitude_diff + 1e-6), 1.0)
        
        # Compute reward based on direction match
        if pred_direction == actual_direction:
            # Correct direction: reward is +1.0 minus log penalty for magnitude error
            reward = 1.0 - log_penalty
        else:
            # Incorrect direction: penalty is -1.0 minus log penalty for magnitude error
            reward = -1.0 - log_penalty
        
        # Scale reward and add exploration bonus
        reward = reward * self.reward_scale
        reward += 0.01  # Small exploration bonus
        
        if debug:
            print(f"[REWARD DEBUG] pred_cum={pred_cumulative:.3f}({pred_direction}), actual_cum={actual_cumulative:.3f}({actual_direction})")
            print(f"[REWARD DEBUG] magnitude_diff={magnitude_diff:.3f}, log_penalty={log_penalty:.3f}, final_reward={reward:.3f}")
        
        return reward
    
    def detect_policy_collapse(self, predictions: torch.Tensor) -> bool:
        """
        Detect if the policy has collapsed to constant predictions.
        
        Args:
            predictions: Recent model predictions
            
        Returns:
            True if policy collapse detected
        """
        # Convert predictions to numpy and track
        pred_values = predictions.cpu().numpy().flatten()
        self.recent_predictions.extend(pred_values)
        
        if len(self.recent_predictions) < 30:  # Need enough samples
            return False
        
        # Check if predictions have very low variance (indicating collapse)
        pred_std = np.std(self.recent_predictions)
        
        if pred_std < self.collapse_threshold:
            print(f"⚠️  Policy collapse detected! Prediction std: {pred_std:.6f}")
            return True
        
        return False
    
    def recover_from_collapse(self):
        """
        Recover from policy collapse by resetting certain parameters and adding noise.
        """
        print(f"🔄 Attempting policy collapse recovery...")
        
        # Clear recent predictions history
        self.recent_predictions.clear()
        
        # Increase entropy coefficient temporarily
        old_entropy_coeff = self.entropy_coeff
        self.entropy_coeff = min(0.15, self.entropy_coeff * 3.0)  # Triple entropy, cap at 0.15
        print(f"  Entropy coefficient: {old_entropy_coeff:.3f} → {self.entropy_coeff:.3f}")
        
        # Reset learning rates to higher values temporarily
        for param_group in self.policy_optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = min(1e-4, old_lr * 2.0)  # Double LR, cap at 1e-4
            print(f"  Policy LR: {old_lr:.6f} → {param_group['lr']:.6f}")
        
        for param_group in self.value_optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = min(1e-3, old_lr * 2.0)  # Double LR, cap at 1e-3
            print(f"  Value LR: {old_lr:.6f} → {param_group['lr']:.6f}")
        
        # Add noise to model parameters to break symmetry
        with torch.no_grad():
            noise_scale = 1e-4
            for param in self.model.parameters():
                if param.dim() > 1:  # Only add noise to weight matrices, not biases
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)
        
        print(f"  Added noise (scale={noise_scale}) to model parameters")
        
        # Clear experience buffer to start fresh
        self.buffer.clear()
        print(f"  Cleared experience buffer")
        
        print(f"🔄 Policy collapse recovery complete")
        
        # Schedule to reduce entropy back after some episodes
        self.recovery_episodes_remaining = 20  # Reduce entropy over next 20 episodes
        self.target_entropy_coeff = old_entropy_coeff
        
    def update_recovery_schedule(self):
        """Update recovery schedule after collapse recovery."""
        if hasattr(self, 'recovery_episodes_remaining') and self.recovery_episodes_remaining > 0:
            self.recovery_episodes_remaining -= 1
            
            # Gradually reduce entropy back to original value
            if self.recovery_episodes_remaining == 0:
                self.entropy_coeff = self.target_entropy_coeff
                print(f"🔄 Recovery complete - entropy reset to {self.entropy_coeff:.3f}")
            else:
                # Linear interpolation back to target
                progress = (20 - self.recovery_episodes_remaining) / 20.0
                self.entropy_coeff = self.entropy_coeff * (1 - progress) + self.target_entropy_coeff * progress
    
    def get_value_estimate(self, state: torch.Tensor) -> float:
        """Get value estimate for a state."""
        try:
            with torch.no_grad():
                value = self.value_net(state.float())
                return value.item()
        except Exception as e:
            print(f"⚠️  Warning: Error getting value estimate: {e}")
            return 0.0
    
    def compute_direction_accuracy(self, predicted_tokens: torch.Tensor, actual_tokens: torch.Tensor) -> float:
        """Compute the accuracy of direction predictions."""
        try:
            # Decode tokens to get close_bucket values
            pred_df = decode_data(predicted_tokens)
            actual_df = decode_data(actual_tokens)
            
            # Convert close_bucket tokens to standardized values
            def token_to_std_value(token):
                try:
                    idx = int(token - StockData.CLOSE_LABELS.min())
                    if 0 <= idx < len(StockData.BIN_VALUES):
                        return StockData.BIN_VALUES[idx]
                    else:
                        return 0.0
                except:
                    return 0.0
            
            # Get close values and compute cumulative movement for each day
            if isinstance(pred_df, pd.DataFrame) and 'close_bucket' in pred_df.columns:
                pred_close_tokens = pred_df['close_bucket'].values
                pred_std_values = [token_to_std_value(token) for token in pred_close_tokens]
            else:
                pred_std_values = []
            
            if isinstance(actual_df, pd.DataFrame) and 'close_bucket' in actual_df.columns:
                actual_close_tokens = actual_df['close_bucket'].values
                actual_std_values = [token_to_std_value(token) for token in actual_close_tokens]
            else:
                actual_std_values = []
            
            # Compute daily directions
            tokens_per_day = len(data_columns)
            correct_directions = 0
            total_days = 0
            
            for day in range(0, min(len(pred_std_values), len(actual_std_values)), tokens_per_day):
                if day + tokens_per_day <= len(pred_std_values) and day + tokens_per_day <= len(actual_std_values):
                    pred_day_values = pred_std_values[day:day + tokens_per_day]
                    actual_day_values = actual_std_values[day:day + tokens_per_day]
                    
                    pred_cumulative = sum(pred_day_values)
                    actual_cumulative = sum(actual_day_values)
                    
                    pred_direction = 1 if pred_cumulative > 0 else (-1 if pred_cumulative < 0 else 0)
                    actual_direction = 1 if actual_cumulative > 0 else (-1 if actual_cumulative < 0 else 0)
                    
                    if pred_direction == actual_direction:
                        correct_directions += 1
                    total_days += 1
            
            return correct_directions / total_days if total_days > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️  Warning: Error computing direction accuracy: {e}")
            return 0.0
    
    def generate_episode(self, 
                        context_tokens: torch.Tensor,
                        target_tokens: torch.Tensor,
                        temperature: float = 0.8,
                        debug: bool = False) -> Tuple[List[float], Dict]:
        """
        Generate an episode by rolling out the policy for 20 days.
        Returns episode rewards and statistics.
        """
        if debug:
            print(f"[EPISODE DEBUG] Context length: {len(context_tokens)}, Target length: {len(target_tokens)}")
            print(f"[EPISODE DEBUG] Starting episode generation with temperature: {temperature}")
        
        # Initialize sequence with context
        current_sequence = context_tokens.clone()
        episode_rewards = []
        episode_actions = []
        episode_log_probs = []
        episode_values = []
        
        # Rollout for 20 days (20 * 9 tokens per day)
        rollout_length = 20 * len(data_columns)
        
        if debug:
            print(f"[EPISODE DEBUG] Rolling out for {rollout_length} tokens ({20} days)")
        
        for step in range(rollout_length):
            # Ensure sequence fits within block size
            if len(current_sequence) > self.model.config.block_size:
                current_sequence = current_sequence[-self.model.config.block_size + 1:]
            
            # Get action distribution from policy
            with torch.no_grad():
                logits = self.model(current_sequence.unsqueeze(0))[0, -1, :]
                probs = F.softmax(logits / temperature, dim=-1)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                value = self.get_value_estimate(current_sequence)
            
            # Store step information
            episode_actions.append(action.item())
            episode_log_probs.append(log_prob.item())
            episode_values.append(value)
            
            if debug and step % 9 == 0:  # Print every day (9 tokens)
                day_num = step // 9 + 1
                print(f"[EPISODE DEBUG] Day {day_num}: action={action.item()}, log_prob={log_prob.item():.3f}, value={value:.3f}")
            
            # Add action to sequence
            current_sequence = torch.cat([current_sequence, action.detach()])
            
            # Compute reward for this step (compare with target)
            if step < len(target_tokens):
                step_reward = self.compute_continuous_reward(
                    current_sequence[-len(data_columns):], 
                    target_tokens[step:step + len(data_columns)],
                    debug=debug
                )
                episode_rewards.append(step_reward)
            else:
                # If we've exceeded target length, use a default reward
                episode_rewards.append(0.0)
            
            # Add experience to buffer for PPO training
            done = (step == rollout_length - 1)
            self.buffer.add(
                state=current_sequence.clone(),
                action=action.item(),
                reward=episode_rewards[-1],
                value=value,
                log_prob=log_prob.item(),
                done=done
            )
        
        # Compute final value for GAE
        final_value = self.get_value_estimate(current_sequence)
        
        # Compute advantages using GAE
        self.buffer.compute_advantages(final_value, self.gamma, self.gae_lambda)
        
        # Compute episode statistics
        total_reward = sum(episode_rewards)
        direction_accuracy = self.compute_direction_accuracy(current_sequence[-rollout_length:], target_tokens)
        
        # Check for policy collapse
        collapse_detected = self.detect_policy_collapse(current_sequence[-rollout_length:])
        
        episode_stats = {
            'total_reward': total_reward,
            'direction_accuracy': direction_accuracy,
            'collapse_detected': collapse_detected,
            'prediction_std': torch.std(current_sequence[-rollout_length:].float()).item(),
            'episode_length': rollout_length
        }
        
        if debug:
            print(f"[EPISODE DEBUG] Episode complete: total_reward={total_reward:.3f}, accuracy={direction_accuracy:.3f}")
            print(f"[EPISODE DEBUG] Collapse detected: {collapse_detected}")
            print(f"[EPISODE DEBUG] Buffer size after episode: {self.buffer.size()}")
        
        return episode_rewards, episode_stats
    
    def update_policy(self, debug: bool = False):
        """
        Update policy using PPO algorithm.
        Returns update statistics.
        """
        if self.buffer.size() < self.batch_size:
            return None
        
        if debug:
            print(f"[POLICY DEBUG] Starting PPO update with buffer size: {self.buffer.size()}")
        
        # Get batch of experiences
        batch_states, batch_actions, batch_returns, batch_advantages, batch_old_log_probs, batch_values = self.buffer.get_batch(self.batch_size)
        
        if debug:
            print(f"[POLICY DEBUG] Batch sizes - states: {len(batch_states)}, actions: {len(batch_actions)}")
            print(f"[POLICY DEBUG] Returns range: [{min(batch_returns):.3f}, {max(batch_returns):.3f}]")
            print(f"[POLICY DEBUG] Advantages range: [{min(batch_advantages):.3f}, {max(batch_advantages):.3f}]")
        
        # Convert to tensors
        batch_states = torch.stack(batch_states).to(self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(self.device)
        batch_returns = torch.tensor(batch_returns, dtype=torch.float32).to(self.device)
        batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32).to(self.device)
        batch_old_log_probs = torch.tensor(batch_old_log_probs, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        
        if debug:
            print(f"[POLICY DEBUG] Normalized advantages - mean: {batch_advantages.mean():.6f}, std: {batch_advantages.std():.6f}")
        
        # Multiple epochs of updates
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.epochs_per_update):
            if debug:
                print(f"[POLICY DEBUG] Epoch {epoch + 1}/{self.epochs_per_update}")
            
            # Forward pass
            logits = self.model(batch_states)[:, -1, :]  # Get logits for last position
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # Compute new log probs and entropy
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # Compute value estimates
            values = self.value_net(batch_states.float()).squeeze()
            
            # Compute policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, batch_returns)
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_loss_coeff * value_loss - 
                         self.entropy_coeff * entropy)
            
            if debug:
                print(f"[POLICY DEBUG] Losses - policy: {policy_loss.item():.6f}, value: {value_loss.item():.6f}, entropy: {entropy.item():.6f}")
            
            # Backward pass
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            
            if debug:
                print(f"[POLICY DEBUG] Gradient norms - policy: {policy_grad_norm:.6f}, value: {value_grad_norm:.6f}")
            
            # Update parameters
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # Store losses
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
        
        # Clear buffer after update
        self.buffer.clear()
        
        # Return statistics
        update_stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'policy_grad_norm': policy_grad_norm.item(),
            'value_grad_norm': value_grad_norm.item()
        }
        
        if debug:
            print(f"[POLICY DEBUG] Update complete - avg policy loss: {update_stats['policy_loss']:.6f}")
            print(f"[POLICY DEBUG] Buffer cleared, size: {self.buffer.size()}")
        
        return update_stats
    
    def train(self, 
              ticker: str,
              data_dir: str,
              num_episodes: int = 200,
              save_every: int = 20,
              out_dir: str = 'out',
              temperature: float = 0.8,
              train_cutoff_date: str = '2022-12-31',
              debug: bool = True):
        """
        Main PPO training loop.
        
        Args:
            ticker: Stock ticker symbol
            data_dir: Directory containing data
            num_episodes: Number of training episodes
            save_every: Save model every N episodes
            out_dir: Output directory
            temperature: Sampling temperature
            train_cutoff_date: Only use data up to this date for training
            debug: Whether to print debug information
        """
        print(f"🚀 Starting PPO training for {ticker}")
        print(f"📚 Training for {num_episodes} episodes")
        
        # Load data
        all_data_df = get_data_for_eval(ticker, data_dir)
        print(f"📊 Loaded {len(all_data_df)} days of data")
        
        # --- DIAGNOSTICS: Print up/down/flat distribution ---
        if 'Close' in all_data_df.columns:
            close_moves = all_data_df['Close'].diff().fillna(0)
            up = (close_moves > 0).sum()
            down = (close_moves < 0).sum()
            flat = (close_moves == 0).sum()
            print(f"[DIAG] Up days: {up}, Down days: {down}, Flat days: {flat}")
        else:
            print("[DIAG] Could not find 'Close' column for up/down/flat diagnostics.")
        
        # --- DIAGNOSTICS: Print average reward for always predicting up, down, random ---
        def fake_pred_tokens(direction: int | str, actual_tokens: torch.Tensor) -> torch.Tensor:
            # direction: 1=up, -1=down, 0=flat, 'random'=random
            tokens = actual_tokens.clone().detach()
            tokens = tokens.to(torch.long)
            if direction == 'random':
                for i in range(0, len(tokens), len(data_columns)):
                    move = int(np.random.choice([-1, 0, 1]))
                    tokens[i:i+len(data_columns)] = tokens[i:i+len(data_columns)] + move
            else:
                move = int(direction)
                for i in range(0, len(tokens), len(data_columns)):
                    tokens[i:i+len(data_columns)] = tokens[i:i+len(data_columns)] + move
            return tokens
        
        # Use first 20 days for diagnostics
        diag_target_df = all_data_df.head(20)
        if isinstance(diag_target_df, pd.DataFrame) and len(diag_target_df) >= 1:
            diag_actual_tokens = encode_data(diag_target_df).squeeze().to(self.device)
            for label, direction in [('Always Up', 1), ('Always Down', -1), ('Always Flat', 0), ('Random', 'random')]:
                fake_pred = fake_pred_tokens(direction, diag_actual_tokens)
                reward = self.compute_continuous_reward(fake_pred, diag_actual_tokens, debug=False)
                print(f"[DIAG] Avg reward for {label}: {reward:.4f}")
            
            # --- DIAGNOSTICS: Print reward for sample up, down, flat predictions ---
            print("[DIAG] Sample reward for up, down, flat predictions:")
            for move, label in [(1, 'Up'), (-1, 'Down'), (0, 'Flat')]:
                fake_pred = fake_pred_tokens(move, diag_actual_tokens)
                reward = self.compute_continuous_reward(fake_pred, diag_actual_tokens, debug=True)
                print(f"[DIAG] Reward for {label}: {reward:.4f}")
        else:
            print("[DIAG] Not enough data for reward diagnostics.")
        
        # Split data: only use data up to train_cutoff_date for training
        train_cutoff_date_obj = datetime.strptime(train_cutoff_date, '%Y-%m-%d').date()
        train_data_df = all_data_df[all_data_df['Date'] <= train_cutoff_date_obj].copy()
        
        print(f"📈 Training data: {len(train_data_df)} days (up to {train_cutoff_date})")
        print(f"📊 Total data: {len(all_data_df)} days")
        print(f"🧪 Test data: {len(all_data_df) - len(train_data_df)} days (from {train_cutoff_date} onwards)")
        
        # Training parameters
        min_context_days = 50
        predict_days = 20
        
        best_accuracy = 0.0
        recent_rewards = deque(maxlen=10)
        collapse_count = 0  # Track number of collapses
        
        for episode in range(num_episodes):
            # Update recovery schedule if in recovery mode
            self.update_recovery_schedule()
            
            if debug and episode % 10 == 0:
                print(f"\n🎯 Episode {episode}/{num_episodes}")
            
            # Sample training window (only from training data)
            max_start_idx = len(train_data_df) - min_context_days - predict_days
            if max_start_idx <= 0:
                print("❌ Not enough training data for training")
                break
            
            start_idx = random.randint(0, max_start_idx)
            context_end_idx = start_idx + min_context_days
            target_end_idx = context_end_idx + predict_days
            
            # Get context and target data (only from training data)
            context_df = train_data_df.iloc[start_idx:context_end_idx]
            target_df = train_data_df.iloc[context_end_idx:target_end_idx]
            
            # Encode to tokens
            context_tokens = encode_data(context_df).squeeze().to(self.device)
            target_tokens = encode_data(target_df).squeeze().to(self.device)
            
            # Generate episode
            episode_rewards, episode_stats = self.generate_episode(
                context_tokens, target_tokens, temperature,
                debug=(debug and episode % 20 == 0)
            )
            
            # Handle policy collapse
            if episode_stats.get('collapse_detected', False):
                collapse_count += 1
                print(f"💥 Policy collapse #{collapse_count} detected at episode {episode}")
                print(f"   Prediction std: {episode_stats.get('prediction_std', 0):.6f}")
                
                # Attempt recovery if not too many collapses
                if collapse_count <= 3:  # Allow up to 3 recovery attempts
                    self.recover_from_collapse()
                else:
                    print(f"⚠️  Too many policy collapses ({collapse_count}). Consider adjusting hyperparameters.")
            
            # Track statistics
            self.episode_rewards.append(episode_stats['total_reward'])
            self.prediction_accuracy.append(episode_stats['direction_accuracy'])
            recent_rewards.append(episode_stats['total_reward'])
            
            # Update policy when buffer is full (skip if just recovered from collapse)
            if self.buffer.size() >= self.batch_size and not episode_stats.get('collapse_detected', False):
                update_stats = self.update_policy(debug=(debug and episode % 20 == 0))
                
                if update_stats:
                    self.policy_losses.append(update_stats['policy_loss'])
                    self.value_losses.append(update_stats['value_loss'])
                    self.entropy_losses.append(update_stats['entropy_loss'])
            
            # Print progress
            if debug and episode % 10 == 0:
                recent_accuracy = np.mean(self.prediction_accuracy[-10:]) if len(self.prediction_accuracy) >= 10 else episode_stats['direction_accuracy']
                recent_reward = np.mean(recent_rewards)
                print(f"📈 Episode {episode}: accuracy={recent_accuracy:.3f}, reward={recent_reward:.3f}")
            
            # Save model periodically
            if episode % save_every == 0 and episode > 0:
                accuracy = np.mean(self.prediction_accuracy[-save_every:])
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    filename = f'ppo_model_episode_{episode}_acc_{accuracy:.3f}.pt'
                    self.save_model(out_dir, filename)
                    print(f"💾 Saved PPO model at episode {episode} with accuracy {accuracy:.3f}")
        
        # Save final model
        final_accuracy = np.mean(self.prediction_accuracy[-10:]) if len(self.prediction_accuracy) >= 10 else 0.0
        final_filename = f'ppo_model_final_acc_{final_accuracy:.3f}.pt'
        self.save_model(out_dir, final_filename)
        print(f"🎉 PPO training complete! Final accuracy: {final_accuracy:.3f}")
        
        # Print summary
        self.print_training_summary()
    
    def save_model(self, out_dir: str, filename: str):
        """Save PPO model checkpoint."""
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        checkpoint = {
            'model': self.model.state_dict(),  # For compatibility with predict.py
            'model_state_dict': self.model.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'prediction_accuracy': self.prediction_accuracy,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'model_args': {
                'n_layer': self.model.config.n_layer,
                'n_head': self.model.config.n_head,
                'n_embd': self.model.config.n_embd,
                'block_size': self.model.config.block_size,
                'bias': self.model.config.bias,
                'vocab_size': self.model.config.vocab_size,
                'dropout': self.model.config.dropout,
            },
            'ppo_config': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'epochs_per_update': self.epochs_per_update,
                'batch_size': self.batch_size,
                'value_loss_coeff': self.value_loss_coeff,
                'entropy_coeff': self.entropy_coeff,
            }
        }
        
        torch.save(checkpoint, os.path.join(out_dir, filename))
        print(f"💾 PPO model saved to {out_dir}/{filename}")
    
    def print_training_summary(self):
        """Print training statistics summary."""
        if not self.episode_rewards:
            print("No training data to summarize")
            return
        
        print("\n" + "="*70)
        print("🎯 PPO TRAINING SUMMARY")
        print("="*70)
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Average episode reward: {np.mean(self.episode_rewards):.3f}")
        print(f"Best episode reward: {np.max(self.episode_rewards):.3f}")
        print(f"Average direction accuracy: {np.mean(self.prediction_accuracy):.3f}")
        print(f"Best direction accuracy: {np.max(self.prediction_accuracy):.3f}")
        print(f"Final 10-episode accuracy: {np.mean(self.prediction_accuracy[-10:]):.3f}")
        
        if self.policy_losses:
            print(f"Average policy loss: {np.mean(self.policy_losses):.6f}")
            print(f"Average value loss: {np.mean(self.value_losses):.6f}")
            print(f"Average entropy loss: {np.mean(self.entropy_losses):.6f}")
        
        print("="*70)


def load_ppo_model(device: str, out_dir: str, ckpt_file: str) -> Tuple[GPT, PPOValueNetwork]:
    """Load PPO model checkpoint."""
    ckpt_path = os.path.join(out_dir, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"❌ Can't find checkpoint file: {ckpt_path}")
        return None, None
    
    print(f"📂 Loading PPO model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Load policy model
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load value network
    value_net = PPOValueNetwork(config.n_embd).to(device)
    value_net.load_state_dict(checkpoint['value_net_state_dict'])
    value_net.eval()
    
    # Print training stats
    if 'prediction_accuracy' in checkpoint:
        accuracy = checkpoint['prediction_accuracy']
        print(f"📊 PPO model accuracy: min={min(accuracy):.3f}, max={max(accuracy):.3f}, final={accuracy[-1]:.3f}")
    
    return model, value_net


def evaluate_ppo_model(model: GPT, 
                      value_net: PPOValueNetwork,
                      ticker: str,
                      data_dir: str,
                      device: str,
                      train_cutoff_date: str = '2022-12-31',
                      eval_start_date: str = '2023-01-01',
                      predict_days: int = 20,
                      temperature: float = 0.5,
                      debug: bool = True) -> Dict[str, float]:
    """Evaluate PPO model performance."""
    print(f"🧪 Evaluating PPO model on {ticker}")
    
    # Load data
    all_data_df = get_data_for_eval(ticker, data_dir)
    train_cutoff_date_obj = datetime.strptime(train_cutoff_date, '%Y-%m-%d').date()
    eval_start_date_obj = datetime.strptime(eval_start_date, '%Y-%m-%d').date()
    
    # Split data: use training data for context, evaluation data for testing
    context_df = all_data_df[all_data_df['Date'] <= train_cutoff_date_obj]
    eval_df = all_data_df[all_data_df['Date'] >= eval_start_date_obj]
    
    print(f"📈 Context data: {len(context_df)} days (up to {train_cutoff_date})")
    print(f"🧪 Evaluation data: {len(eval_df)} days (from {eval_start_date} onwards)")
    
    # Use last part of context + first part of evaluation data for prediction
    if len(context_df) < 100:
        print("❌ Not enough context data for evaluation")
        return {}
    
    # Use last 100 days of training data as context
    context_df = context_df.tail(100)
    actual_df = eval_df.head(predict_days)
    
    if len(actual_df) < predict_days:
        print(f"⚠️  Only {len(actual_df)} days of test data available")
        predict_days = len(actual_df)
    
    # Encode context
    context_tokens = encode_data(context_df).to(device)
    
    # Generate predictions
    print(f"🔮 Generating {predict_days} day predictions...")
    tokens_per_day = len(data_columns)
    max_new_tokens = predict_days * tokens_per_day
    
    with torch.no_grad():
        generated = model.generate(context_tokens, max_new_tokens, temperature=temperature)
        predictions = generated[:, -max_new_tokens:]
    
    # Compute reward using PPO trainer's method
    trainer = PPOTrainer(model, device)
    actual_tokens = encode_data(actual_df).squeeze()
    
    reward = trainer.compute_continuous_reward(predictions.squeeze(), actual_tokens, debug=debug)
    
    # Additional metrics
    pred_df = decode_data(predictions)
    def safe_token_to_std(token):
        try:
            idx = int(token - StockData.CLOSE_LABELS.min())
            if 0 <= idx < len(StockData.BIN_VALUES):
                return StockData.BIN_VALUES[idx]
            else:
                return 0.0
        except:
            return 0.0
    if isinstance(pred_df, pd.DataFrame):
        pred_close_buckets = pred_df['close_bucket']
        pred_std_values = np.array([safe_token_to_std(token) for token in pred_close_buckets])
    else:
        pred_std_values = np.array([])
    if isinstance(actual_df, pd.DataFrame):
        actual_close_buckets = actual_df['close_bucket']
        actual_std_values = np.array([safe_token_to_std(token) for token in actual_close_buckets])
    else:
        actual_std_values = np.array([])
    pred_cumulative = float(np.sum(pred_std_values))
    actual_cumulative = float(np.sum(actual_std_values))
    
    # Use same 3-way direction logic as training
    if pred_cumulative > 0:
        pred_direction = 1
    elif pred_cumulative < 0:
        pred_direction = -1
    else:
        pred_direction = 0
        
    if actual_cumulative > 0:
        actual_direction = 1
    elif actual_cumulative < 0:
        actual_direction = -1
    else:
        actual_direction = 0
    
    direction_match = pred_direction == actual_direction
    
    metrics = {
        'continuous_reward': reward,
        'predicted_cumulative': pred_cumulative,
        'actual_cumulative': actual_cumulative,
        'predicted_direction': pred_direction,
        'actual_direction': actual_direction,
        'direction_match': direction_match,
        'mae': np.mean(np.abs(pred_std_values - actual_std_values)),
        'mse': np.mean((pred_std_values - actual_std_values)**2)
    }
    
    if debug:
        print(f"\n📊 PPO Evaluation Results:")
        print(f"  Continuous reward: {metrics['continuous_reward']:.3f}")
        print(f"  Predicted cumulative: {metrics['predicted_cumulative']:.3f} (direction: {metrics['predicted_direction']})")
        print(f"  Actual cumulative: {metrics['actual_cumulative']:.3f} (direction: {metrics['actual_direction']})")
        print(f"  Direction match: {metrics['direction_match']}")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  MSE: {metrics['mse']:.3f}")
    
    return metrics


def main():
    """Main function for PPO training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO training for stock prediction')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--model_file', type=str, default='ckpt.pt', help='Base model checkpoint')
    parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-5, help='Policy learning rate')
    parser.add_argument('--value_lr', type=float, default=1e-4, help='Value learning rate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='mps', help='Device (cpu/cuda/mps)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model instead of training')
    parser.add_argument('--ppo_model', type=str, help='PPO model checkpoint for evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs_per_update', type=int, default=4, help='Epochs per update')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--train_cutoff_date', type=str, default='2022-12-31', help='Training data cutoff date')
    parser.add_argument('--eval_start_date', type=str, default='2023-01-01', help='Evaluation data start date')
    parser.add_argument('--debug', action='store_true', help='Enable debug output during training')
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_dir, args.data_dir)
    
    if args.evaluate:
        # Evaluate mode
        if args.ppo_model:
            model, value_net = load_ppo_model(args.device, args.out_dir, args.ppo_model)
            if model and value_net:
                metrics = evaluate_ppo_model(
                    model, value_net, args.ticker, data_path, args.device,
                    train_cutoff_date=args.train_cutoff_date,
                    eval_start_date=args.eval_start_date,
                    debug=args.debug
                )
                print(f"🎯 Final PPO evaluation: {metrics}")
        else:
            print("❌ Please specify --ppo_model for evaluation")
    else:
        # Training mode
        print(f"🚀 Starting PPO training with {args.episodes} episodes")
        
        # Load base model
        model = load_model(args.device, args.out_dir, args.model_file)
        if model is None:
            print("❌ Failed to load base model")
            return
        
        # Create PPO trainer
        trainer = PPOTrainer(
            model=model,
            device=args.device,
            learning_rate=args.lr,
            value_lr=args.value_lr,
            batch_size=args.batch_size,
            epochs_per_update=args.epochs_per_update,
            clip_epsilon=args.clip_epsilon
        )
        
        # Train
        trainer.train(
            ticker=args.ticker,
            data_dir=data_path,
            num_episodes=args.episodes,
            temperature=args.temperature,
            train_cutoff_date=args.train_cutoff_date,
            out_dir=args.out_dir,
            debug=args.debug
        )


if __name__ == "__main__":
    main()