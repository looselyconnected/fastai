"""
Reinforcement Learning implementation for stock prediction model.

This module implements RL training to optimize the model for 20-day cumulative direction prediction
rather than just next-token prediction. The reward function is based on whether the predicted
20-day cumulative movement direction matches the actual direction.

Key components:
- RLTrainer: Main RL training class using policy gradient
- compute_20_day_reward: Reward function based on cumulative direction matching
- generate_episodes: Generate prediction episodes for RL training
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Tuple, Dict, Optional
import random
import logging
import glob
import argparse
import json
from collections import defaultdict, deque

from model import GPT, load_model
from data import data_columns, get_data_for_eval, decode_data, encode_data
from stockdata import StockData

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLTrainer:
    """
    Reinforcement Learning trainer for stock prediction model.
    
    Uses policy gradient (REINFORCE) to optimize model for 20-day cumulative direction prediction.
    The model generates sequences of tokens representing stock movements, and gets rewards
    based on whether the cumulative direction matches reality.
    """
    
    def __init__(self, 
                 model: GPT, 
                 device: str = 'cpu',
                 learning_rate: float = 1e-5,
                 gamma: float = 0.99,
                 baseline_momentum: float = 0.9,
                 entropy_coeff: float = 0.01):
        """
        Initialize RL trainer.
        
        Args:
            model: Pre-trained GPT model
            device: Device to run on ('cpu', 'cuda', 'mps')
            learning_rate: Learning rate for RL optimizer
            gamma: Discount factor for future rewards
            baseline_momentum: Momentum for baseline (value function) updates
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.baseline_momentum = baseline_momentum
        self.entropy_coeff = entropy_coeff
        
        # Create separate optimizer for RL updates
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Running baseline for reward normalization (simple exponential average)
        self.baseline = 0.0
        
        # Track training statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.prediction_accuracy = []
        
        print(f"✓ RLTrainer initialized with lr={learning_rate}, gamma={gamma}")
        print(f"✓ Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    def compute_20_day_reward(self, 
                            predicted_tokens: torch.Tensor, 
                            actual_tokens: torch.Tensor,
                            debug: bool = False) -> float:
        """
        Compute reward based on 20-day cumulative direction matching.
        
        Args:
            predicted_tokens: Model predictions (shape: [sequence_length])
            actual_tokens: Ground truth tokens (shape: [sequence_length])
            debug: Whether to print debug information
            
        Returns:
            Reward value (1.0 if directions match, -1.0 if they don't)
        """
        # Ensure tokens are properly shaped for decoding
        if len(predicted_tokens) % len(data_columns) != 0:
            # Pad or truncate to align with data columns
            target_length = (len(predicted_tokens) // len(data_columns)) * len(data_columns)
            predicted_tokens = predicted_tokens[:target_length]
        
        if len(actual_tokens) % len(data_columns) != 0:
            target_length = (len(actual_tokens) // len(data_columns)) * len(data_columns)
            actual_tokens = actual_tokens[:target_length]
        
        # Decode tokens to dataframe format
        pred_df = decode_data(predicted_tokens.unsqueeze(0))
        actual_df = decode_data(actual_tokens.unsqueeze(0))
        
        # Extract close_bucket values (main price movement indicator)
        pred_close = pred_df['close_bucket'].values
        actual_close = actual_df['close_bucket'].values
        
        # Convert tokens to standard deviation values using StockData mapping
        # Handle invalid tokens by clipping to valid range
        def token_to_std_value(token):
            try:
                idx = int(token - StockData.CLOSE_LABELS.min())
                if 0 <= idx < len(StockData.BIN_VALUES):
                    return StockData.BIN_VALUES[idx]
                else:
                    # Invalid token, return neutral value
                    return 0.0
            except:
                return 0.0
        
        pred_std_values = np.array([token_to_std_value(token) for token in pred_close])
        actual_std_values = np.array([token_to_std_value(token) for token in actual_close])
        
        # Compute cumulative movements
        pred_cumulative = np.sum(pred_std_values)
        actual_cumulative = np.sum(actual_std_values)
        
        # Determine directions (1 = up, -1 = down, 0 = flat)
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
        
        # Reward is 1 if directions match, -1 if they don't
        # reward = 1.0 if pred_direction == actual_direction else -1.0
        if pred_direction == actual_direction:
            reward = 1.0
        else:
            # similar to when writing an option, the reward for getting it wrong is the magnitude of the difference
            # minus 1.0 (proxy for the price of the option)
            reward = 1.0 - abs(actual_cumulative)

        if debug:
            print(f"  Predicted cumulative: {pred_cumulative:.3f} (direction: {pred_direction})")
            print(f"  Actual cumulative: {actual_cumulative:.3f} (direction: {actual_direction})")
            print(f"  Reward: {reward}")
        
        return reward
    
    def generate_episode(self, 
                        context_tokens: torch.Tensor,
                        target_tokens: torch.Tensor,
                        temperature: float = 0.8,
                        debug: bool = False) -> Tuple[torch.Tensor, List[float], List[torch.Tensor]]:
        """
        Generate a single episode (20-day prediction sequence).
        
        Args:
            context_tokens: Historical context tokens
            target_tokens: Ground truth tokens for reward computation
            temperature: Sampling temperature
            debug: Whether to print debug information
            
        Returns:
            Tuple of (generated_sequence, rewards, log_probs)
        """
        if debug:
            print(f"🎯 Generating episode with context length: {len(context_tokens)}")
        
        # Generate 20 days worth of tokens
        predict_days = 20
        tokens_per_day = len(data_columns)  # 9 tokens per day
        max_new_tokens = predict_days * tokens_per_day
        
        # Store log probabilities for policy gradient
        log_probs = []
        rewards = []
        
        # Generate sequence token by token to capture log probabilities
        current_sequence = context_tokens.clone()
        generated_tokens_list = []  # Track generated tokens separately
        
        for step in range(max_new_tokens):
            # Ensure we don't exceed block size
            if len(current_sequence) >= self.model.config.block_size:
                # Crop the sequence to fit in block size, keeping the most recent tokens
                # current_sequence = current_sequence[-self.model.config.block_size + 1:]
                current_sequence = current_sequence[-self.model.config.block_size:]
            
            # Get model predictions (need gradients for policy gradient)
            logits, _ = self.model(current_sequence.unsqueeze(0))
            logits = logits[0, -1, :] / temperature  # Get last token logits
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            sampled_token = torch.multinomial(probs, 1)
            
            # Store log probability
            log_prob = F.log_softmax(logits, dim=-1)[sampled_token]
            log_probs.append(log_prob)
            
            # Store generated token separately
            generated_tokens_list.append(sampled_token.detach())
            
            # Append to sequence (detach to avoid accumulating gradients through the sequence)
            current_sequence = torch.cat([current_sequence, sampled_token.detach()])
        
        # Convert generated tokens list to tensor
        generated_tokens = torch.cat(generated_tokens_list)
        
        # Compute single reward for entire episode based on final cumulative outcome
        episode_reward = self.compute_20_day_reward(generated_tokens, target_tokens, debug=debug)
        rewards = [episode_reward]  # Single reward for the entire episode
        
        if debug:
            print(f"📊 Episode stats: Single episode reward: {episode_reward:.3f}")
        
        return generated_tokens, rewards, log_probs
    
    def compute_discounted_rewards(self, rewards: List[float]) -> List[float]:
        """
        Compute discounted cumulative rewards.
        
        Args:
            rewards: List of immediate rewards
            
        Returns:
            List of discounted cumulative rewards
        """
        discounted = []
        cumulative = 0.0
        
        # Compute discounted rewards backward
        for reward in reversed(rewards):
            cumulative = reward + self.gamma * cumulative
            discounted.append(cumulative)
        
        return list(reversed(discounted))
    
    def train_episode(self, 
                     context_tokens: torch.Tensor,
                     target_tokens: torch.Tensor,
                     temperature: float = 0.8,
                     debug: bool = False) -> Dict[str, float]:
        """
        Train on a single episode using policy gradient.
        
        Args:
            context_tokens: Historical context tokens
            target_tokens: Ground truth tokens
            temperature: Sampling temperature
            debug: Whether to print debug information
            
        Returns:
            Dictionary with training statistics
        """
        # Generate episode
        generated_tokens, rewards, log_probs = self.generate_episode(
            context_tokens, target_tokens, temperature, debug
        )
        
        # Single reward for entire episode (no discounting needed)
        episode_reward = rewards[0]  # Only one reward now
        
        # Update baseline (simple exponential moving average)
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * episode_reward
        
        # Compute policy gradient loss with entropy regularization
        # Apply the same reward to all tokens in the sequence
        advantage = episode_reward - self.baseline
        
        policy_loss = 0.0
        entropy_loss = 0.0
        
        for log_prob in log_probs:
            # Each token gets the same advantage (entire episode reward)
            policy_loss += -log_prob * advantage
            
            # Add entropy regularization to encourage exploration
            entropy_loss += -log_prob  # Entropy of the policy
        
        # Average over sequence length
        policy_loss = policy_loss / len(log_probs)
        entropy_loss = entropy_loss / len(log_probs)
        
        # Combine policy loss with entropy regularization
        total_loss = policy_loss - self.entropy_coeff * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        
        self.optimizer.step()
        
        # Track statistics
        self.episode_rewards.append(episode_reward)
        self.episode_losses.append(total_loss.item())
        
        # Direction accuracy is simply whether the episode reward is positive
        direction_correct = 1.0 if episode_reward > 0 else 0.0
        self.prediction_accuracy.append(direction_correct)
        
        stats = {
            'episode_return': episode_reward,
            'policy_loss': policy_loss.item(),
            'total_loss': total_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'direction_accuracy': direction_correct,
            'baseline': self.baseline,
            'episode_reward': episode_reward
        }
        
        if debug:
            print(f"📈 Training stats: {stats}")
        
        return stats
    
    def train(self, 
              ticker: str,
              data_dir: str,
              num_episodes: int = 100,
              save_every: int = 10,
              out_dir: str = 'out',
              temperature: float = 0.8,
              train_cutoff_date: str = '2022-12-31',
              debug: bool = True):
        """
        Main training loop.
        
        Args:
            ticker: Stock ticker symbol
            data_dir: Directory containing data
            num_episodes: Number of training episodes
            save_every: Save model every N episodes
            out_dir: Output directory for saving models
            temperature: Sampling temperature
            debug: Whether to print debug information
        """
        print(f"🚀 Starting RL training for {ticker}")
        print(f"📚 Training for {num_episodes} episodes")
        
        # Load data
        all_data_df = get_data_for_eval(ticker, data_dir)
        print(f"📊 Loaded {len(all_data_df)} days of data")
        
        # Split data: only use data up to train_cutoff_date for training
        train_cutoff_date_obj = datetime.strptime(train_cutoff_date, '%Y-%m-%d').date()
        train_data_df = all_data_df[all_data_df['Date'] <= train_cutoff_date_obj].copy()
        
        print(f"📈 Training data: {len(train_data_df)} days (up to {train_cutoff_date})")
        print(f"📊 Total data: {len(all_data_df)} days")
        print(f"🧪 Test data: {len(all_data_df) - len(train_data_df)} days (from {train_cutoff_date} onwards)")
        
        # Split data for training episodes
        # Calculate context length based on model's block size
        tokens_per_day = len(data_columns)  # 9 tokens per day
        max_context_tokens = self.model.config.block_size - (20 * tokens_per_day)  # Leave room for prediction
        min_context_days = max(50, max_context_tokens // tokens_per_day)  # At least 50 days
        predict_days = 20
        
        best_accuracy = 0.0
        
        print(f"🔧 Model block size: {self.model.config.block_size}")
        print(f"🔧 Tokens per day: {tokens_per_day}")
        print(f"🔧 Min context days: {min_context_days}")
        print(f"🔧 Predict days: {predict_days}")
        
        for episode in range(num_episodes):
            if debug and episode % 10 == 0:
                print(f"\n🎯 Episode {episode}/{num_episodes}")
            
            # Randomly sample a training window (only from training data)
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
            context_tokens = encode_data(context_df).squeeze()
            target_tokens = encode_data(target_df).squeeze()
            
            # Train on this episode
            stats = self.train_episode(
                context_tokens.to(self.device),
                target_tokens.to(self.device),
                temperature=temperature,
                debug=debug and episode % 20 == 0
            )
            
            # Print progress
            if debug and episode % 10 == 0:
                recent_accuracy = np.mean(self.prediction_accuracy[-10:]) if len(self.prediction_accuracy) >= 10 else stats['direction_accuracy']
                recent_return = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else stats['episode_return']
                print(f"📈 Episode {episode}: accuracy={recent_accuracy:.3f}, return={recent_return:.3f}, loss={stats['policy_loss']:.6f}")
            
            # Save model periodically
            if episode % save_every == 0 and episode > 0:
                accuracy = np.mean(self.prediction_accuracy[-save_every:])
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.save_model(out_dir, f'rl_model_episode_{episode}_acc_{accuracy:.3f}.pt')
                    print(f"💾 Saved model at episode {episode} with accuracy {accuracy:.3f}")
        
        # Save final model
        final_accuracy = np.mean(self.prediction_accuracy[-10:]) if len(self.prediction_accuracy) >= 10 else 0.0
        self.save_model(out_dir, f'rl_model_final_acc_{final_accuracy:.3f}.pt')
        print(f"🎉 Training complete! Final accuracy: {final_accuracy:.3f}")
        
        # Print training summary
        self.print_training_summary()
    
    def save_model(self, out_dir: str, filename: str):
        """Save model checkpoint."""
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        checkpoint = {
            'model': self.model.state_dict(),  # Use 'model' key for compatibility with load_model()
            'model_state_dict': self.model.state_dict(),  # Keep for RL loading
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline': self.baseline,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'prediction_accuracy': self.prediction_accuracy,
            'model_args': {
                'n_layer': self.model.config.n_layer,
                'n_head': self.model.config.n_head,
                'n_embd': self.model.config.n_embd,
                'block_size': self.model.config.block_size,
                'bias': self.model.config.bias,
                'vocab_size': self.model.config.vocab_size,
                'dropout': self.model.config.dropout,
            }
        }
        
        torch.save(checkpoint, os.path.join(out_dir, filename))
        print(f"💾 Model saved to {out_dir}/{filename}")
    
    def print_training_summary(self):
        """Print training statistics summary."""
        if not self.episode_rewards:
            print("No training data to summarize")
            return
        
        print("\n" + "="*60)
        print("🎯 RL TRAINING SUMMARY")
        print("="*60)
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Average episode return: {np.mean(self.episode_rewards):.3f}")
        print(f"Best episode return: {np.max(self.episode_rewards):.3f}")
        print(f"Final baseline: {self.baseline:.3f}")
        print(f"Average direction accuracy: {np.mean(self.prediction_accuracy):.3f}")
        print(f"Best direction accuracy: {np.max(self.prediction_accuracy):.3f}")
        print(f"Final 10-episode accuracy: {np.mean(self.prediction_accuracy[-10:]):.3f}")
        print("="*60)


class MultiStockRLTrainer:
    """
    Multi-stock vanilla RL trainer that handles training across multiple stocks.
    """
    
    def __init__(self, 
                 model: GPT,
                 device: str = 'cpu',
                 learning_rate: float = 1e-5,
                 **rl_kwargs):
        """
        Initialize multi-stock RL trainer.
        
        Args:
            model: Pre-trained GPT model
            device: Device to run on
            learning_rate: Learning rate
            **rl_kwargs: Additional RL parameters
        """
        self.device = device
        self.model = model
        
        # Create vanilla RL trainer
        self.rl_trainer = RLTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            **rl_kwargs
        )
        
        # Track multi-stock statistics
        self.stock_performance = defaultdict(list)
        self.global_episode_count = 0
        self.stock_episode_counts = defaultdict(int)
        
        print(f"✓ Multi-Stock RL Trainer initialized")
        print(f"  Learning rate: {learning_rate}")
    
    def get_available_stocks(self, data_dir: str, 
                           exclude_patterns: List[str] = ['^TNX', '^VIX']) -> List[str]:
        """
        Get list of available stocks from data directory.
        
        Args:
            data_dir: Data directory path
            exclude_patterns: Patterns to exclude (e.g., index symbols)
            
        Returns:
            List of stock tickers
        """
        train_files = glob.glob(os.path.join(data_dir, "*_train.csv"))
        stocks = []
        
        for file_path in train_files:
            filename = os.path.basename(file_path)
            ticker = filename.replace('_train.csv', '')
            
            # Exclude certain patterns
            exclude = False
            for pattern in exclude_patterns:
                if pattern in ticker:
                    exclude = True
                    break
            
            if not exclude:
                stocks.append(ticker)
        
        stocks.sort()
        print(f"📊 Found {len(stocks)} stocks for training")
        return stocks
    
    def validate_stock_data(self, 
                          stocks: List[str], 
                          data_dir: str,
                          train_cutoff_date: str,
                          min_training_days: int = 500) -> List[str]:
        """
        Validate stock data and filter out stocks with insufficient data.
        
        Args:
            stocks: List of stock tickers
            data_dir: Data directory
            train_cutoff_date: Training cutoff date
            min_training_days: Minimum required training days
            
        Returns:
            List of valid stock tickers
        """
        valid_stocks = []
        train_cutoff_date_obj = datetime.strptime(train_cutoff_date, '%Y-%m-%d').date()
        
        print(f"🔍 Validating stock data (min {min_training_days} training days)...")
        
        for ticker in stocks:
            try:
                df = get_data_for_eval(ticker, data_dir)
                train_data = df[df['Date'] <= train_cutoff_date_obj]
                
                if len(train_data) >= min_training_days:
                    valid_stocks.append(ticker)
                    print(f"  ✓ {ticker}: {len(train_data)} training days")
                else:
                    print(f"  ✗ {ticker}: {len(train_data)} training days (insufficient)")
                    
            except Exception as e:
                print(f"  ✗ {ticker}: Error loading data - {e}")
        
        print(f"📈 {len(valid_stocks)} stocks validated for training")
        return valid_stocks
    
    def train_multi_stock(self,
                         stocks: List[str],
                         data_dir: str,
                         num_episodes: int = 500,
                         episodes_per_stock: int = 5,
                         save_every: int = 50,
                         eval_every: int = 100,
                         out_dir: str = 'out',
                         train_cutoff_date: str = '2022-12-31',
                         temperature: float = 0.8,
                         debug: bool = True):
        """
        Main multi-stock training loop.
        
        Args:
            stocks: List of stock tickers to train on
            data_dir: Data directory
            num_episodes: Total number of episodes
            episodes_per_stock: Episodes per stock before switching
            save_every: Save model every N episodes
            eval_every: Evaluate on test set every N episodes
            out_dir: Output directory
            train_cutoff_date: Training data cutoff
            temperature: Sampling temperature
            debug: Debug output
        """
        print(f"🚀 Starting Multi-Stock RL Training")
        print(f"📚 {len(stocks)} stocks, {num_episodes} total episodes")
        print(f"🔄 {episodes_per_stock} episodes per stock rotation")
        print(f"📅 Training cutoff: {train_cutoff_date}")
        
        best_avg_accuracy = 0.0
        recent_rewards = deque(maxlen=20)
        stock_rotation_idx = 0
        
        for episode in range(num_episodes):
            # Select current stock (rotate every episodes_per_stock)
            if episode % episodes_per_stock == 0:
                current_stock = stocks[stock_rotation_idx % len(stocks)]
                stock_rotation_idx += 1
                if debug and episode % (episodes_per_stock * 5) == 0:
                    print(f"\n🎯 Episode {episode}/{num_episodes} - Training on {current_stock}")
            
            try:
                # Load stock data
                all_data_df = get_data_for_eval(current_stock, data_dir)
                train_cutoff_date_obj = datetime.strptime(train_cutoff_date, '%Y-%m-%d').date()
                train_data_df = all_data_df[all_data_df['Date'] <= train_cutoff_date_obj].copy()
                
                if len(train_data_df) < 100:
                    print(f"⚠️  Skipping {current_stock} - insufficient training data")
                    continue
                
                # Sample training window
                min_context_days = 80
                predict_days = 20
                max_start_idx = len(train_data_df) - min_context_days - predict_days
                
                if max_start_idx <= 0:
                    print(f"⚠️  Skipping {current_stock} - insufficient data window")
                    continue
                
                start_idx = random.randint(0, max_start_idx)
                context_end_idx = start_idx + min_context_days
                target_end_idx = context_end_idx + predict_days
                
                context_df = train_data_df.iloc[start_idx:context_end_idx]
                target_df = train_data_df.iloc[context_end_idx:target_end_idx]
                
                # Encode to tokens
                context_tokens = encode_data(context_df).squeeze().to(self.device)
                target_tokens = encode_data(target_df).squeeze().to(self.device)
                
                # Train episode
                stats = self.rl_trainer.train_episode(
                    context_tokens, target_tokens, temperature,
                    debug=(debug and episode % 50 == 0)
                )
                
                # Track per-stock performance
                self.stock_performance[current_stock].append({
                    'episode': episode,
                    'return': stats['episode_return'],
                    'accuracy': stats['direction_accuracy'],
                    'loss': stats['total_loss']
                })
                self.stock_episode_counts[current_stock] += 1
                
                recent_rewards.append(stats['episode_return'])
                
                # Print progress
                if debug and episode % 25 == 0:
                    recent_accuracy = np.mean([ep['accuracy'] for eps in self.stock_performance.values() 
                                             for ep in eps[-5:]])  # Last 5 episodes per stock
                    recent_return = np.mean(recent_rewards) if recent_rewards else 0.0
                    print(f"📈 Episode {episode}: {current_stock} acc={stats['direction_accuracy']:.3f}, "
                          f"avg_acc={recent_accuracy:.3f}, avg_return={recent_return:.3f}")
                
            except Exception as e:
                print(f"❌ Error training on {current_stock} at episode {episode}: {e}")
                continue
            
            # Save model periodically
            if episode % save_every == 0 and episode > 0:
                avg_accuracy = np.mean([ep['accuracy'] for eps in self.stock_performance.values() 
                                      for ep in eps[-10:]])  # Last 10 episodes per stock
                if avg_accuracy > best_avg_accuracy:
                    best_avg_accuracy = avg_accuracy
                    filename = f'multi_stock_rl_episode_{episode}_acc_{avg_accuracy:.3f}.pt'
                    self.rl_trainer.save_model(out_dir, filename)
                    print(f"💾 Saved multi-stock RL model at episode {episode} with accuracy {avg_accuracy:.3f}")
            
            # Comprehensive evaluation
            if episode % eval_every == 0 and episode > 0:
                self.evaluate_on_multiple_stocks(stocks[:5], data_dir, train_cutoff_date, debug=True)
        
        # Save final model
        final_accuracy = np.mean([ep['accuracy'] for eps in self.stock_performance.values() 
                                for ep in eps[-20:]])  # Last 20 episodes per stock
        final_filename = f'multi_stock_rl_final_acc_{final_accuracy:.3f}.pt'
        self.rl_trainer.save_model(out_dir, final_filename)
        
        print(f"🎉 Multi-stock RL training complete! Final accuracy: {final_accuracy:.3f}")
        self.print_training_summary(stocks)
    
    def evaluate_on_multiple_stocks(self,
                                  stocks: List[str],
                                  data_dir: str,
                                  train_cutoff_date: str = '2022-12-31',
                                  eval_start_date: str = '2023-01-01',
                                  debug: bool = False) -> Dict[str, Dict]:
        """
        Evaluate model performance on multiple stocks.
        
        Args:
            stocks: List of stocks to evaluate
            data_dir: Data directory
            train_cutoff_date: Training cutoff date
            eval_start_date: Evaluation start date
            debug: Debug output
            
        Returns:
            Dictionary of evaluation results per stock
        """
        print(f"\n🧪 Evaluating on {len(stocks)} stocks (2023+ data)...")
        
        results = {}
        all_rewards = []
        all_accuracies = []
        
        for ticker in stocks:
            try:
                metrics = evaluate_rl_model(
                    self.model, ticker, data_dir, self.device,
                    cutoff_date=train_cutoff_date,
                    debug=debug
                )
                
                results[ticker] = metrics
                all_rewards.append(metrics.get('direction_reward', 0))
                all_accuracies.append(1.0 if metrics.get('direction_match', False) else 0.0)
                
                if debug:
                    print(f"  {ticker}: reward={metrics.get('direction_reward', 0):.3f}, "
                          f"match={metrics.get('direction_match', False)}")
                
            except Exception as e:
                print(f"  ❌ {ticker}: Evaluation failed - {e}")
                results[ticker] = {'error': str(e)}
        
        # Summary statistics
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        
        print(f"📊 Multi-stock evaluation summary:")
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Direction accuracy: {avg_accuracy:.3f} ({len([a for a in all_accuracies if a > 0])}/{len(all_accuracies)})")
        
        return results
    
    def print_training_summary(self, stocks: List[str]):
        """Print comprehensive training summary."""
        print("\n" + "="*80)
        print("🎯 MULTI-STOCK RL TRAINING SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_episodes = sum(len(eps) for eps in self.stock_performance.values())
        total_returns = [ep['return'] for eps in self.stock_performance.values() for ep in eps]
        total_accuracies = [ep['accuracy'] for eps in self.stock_performance.values() for ep in eps]
        
        print(f"Total episodes: {total_episodes}")
        print(f"Stocks trained: {len(self.stock_performance)}")
        print(f"Average return: {np.mean(total_returns):.3f}")
        print(f"Average accuracy: {np.mean(total_accuracies):.3f}")
        print(f"Best accuracy: {np.max(total_accuracies):.3f}")
        
        # Per-stock summary
        print(f"\nPer-stock performance:")
        for stock in sorted(self.stock_performance.keys()):
            episodes = self.stock_performance[stock]
            if episodes:
                avg_return = np.mean([ep['return'] for ep in episodes])
                avg_accuracy = np.mean([ep['accuracy'] for ep in episodes])
                best_accuracy = np.max([ep['accuracy'] for ep in episodes])
                print(f"  {stock:6}: {len(episodes):3} episodes, "
                      f"avg_return={avg_return:6.2f}, avg_acc={avg_accuracy:.3f}, best_acc={best_accuracy:.3f}")
        
        print("="*80)


def load_rl_model(device: str, out_dir: str, ckpt_file: str) -> GPT:
    """
    Load a model checkpoint saved during RL training.
    
    Args:
        device: Device to load model on
        out_dir: Directory containing checkpoint
        ckpt_file: Checkpoint filename
        
    Returns:
        Loaded GPT model
    """
    ckpt_path = os.path.join(out_dir, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"❌ Can't find checkpoint file: {ckpt_path}")
        return None
    
    print(f"📂 Loading RL model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract model args
    model_args = checkpoint['model_args']
    
    # Create model
    from model import GPTConfig
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Print training stats if available
    if 'prediction_accuracy' in checkpoint:
        accuracy = checkpoint['prediction_accuracy']
        print(f"📊 Model accuracy history: min={min(accuracy):.3f}, max={max(accuracy):.3f}, final={accuracy[-1]:.3f}")
    
    return model


def evaluate_rl_model(model: GPT, 
                     ticker: str,
                     data_dir: str,
                     device: str,
                     cutoff_date: str = '2023-12-04',
                     predict_days: int = 20,
                     temperature: float = 0.5,
                     debug: bool = True) -> Dict[str, float]:
    """
    Evaluate RL model performance on test data.
    
    Args:
        model: Trained RL model
        ticker: Stock ticker
        data_dir: Data directory
        device: Device to run on
        cutoff_date: Date to split train/test
        predict_days: Number of days to predict
        temperature: Sampling temperature
        debug: Whether to print debug info
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"🧪 Evaluating RL model on {ticker}")
    
    # Load data
    all_data_df = get_data_for_eval(ticker, data_dir)
    cutoff_date_obj = datetime.strptime(cutoff_date, '%Y-%m-%d').date()
    
    # Split data
    context_df = all_data_df[all_data_df['Date'] <= cutoff_date_obj]
    actual_df = all_data_df[all_data_df['Date'] > cutoff_date_obj].head(predict_days)
    
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
    
    # Decode predictions
    pred_df = decode_data(predictions)
    
    # Calculate metrics
    trainer = RLTrainer(model, device)  # Just for reward computation
    actual_tokens = encode_data(actual_df).squeeze()
    
    reward = trainer.compute_20_day_reward(predictions.squeeze(), actual_tokens, debug=debug)
    
    # Additional metrics
    def safe_token_to_std(token):
        try:
            idx = int(token - StockData.CLOSE_LABELS.min())
            if 0 <= idx < len(StockData.BIN_VALUES):
                return StockData.BIN_VALUES[idx]
            else:
                return 0.0
        except:
            return 0.0
    
    pred_std_values = pred_df['close_bucket'].apply(safe_token_to_std).values
    actual_std_values = actual_df['close_bucket'].apply(safe_token_to_std).values
    
    pred_cumulative = np.sum(pred_std_values)
    actual_cumulative = np.sum(actual_std_values)
    
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
        'direction_reward': reward,
        'predicted_cumulative': pred_cumulative,
        'actual_cumulative': actual_cumulative,
        'predicted_direction': pred_direction,
        'actual_direction': actual_direction,
        'direction_match': direction_match,
        'mae': np.mean(np.abs(pred_std_values - actual_std_values)),
        'mse': np.mean((pred_std_values - actual_std_values)**2)
    }
    
    if debug:
        print(f"\n📊 Evaluation Results:")
        print(f"  Direction reward: {metrics['direction_reward']}")
        print(f"  Predicted cumulative: {metrics['predicted_cumulative']:.3f} (direction: {metrics['predicted_direction']})")
        print(f"  Actual cumulative: {metrics['actual_cumulative']:.3f} (direction: {metrics['actual_direction']})")
        print(f"  Direction match: {metrics['direction_match']}")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  MSE: {metrics['mse']:.3f}")
    
    return metrics


def main():
    """Main function for RL training."""
    parser = argparse.ArgumentParser(description='RL training for stock prediction')
    
    # Basic parameters
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker (for single stock training)')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--model_file', type=str, default='ckpt.pt', help='Base model checkpoint')
    parser.add_argument('--device', type=str, default='mps', help='Device (cpu/cuda/mps)')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Entropy coefficient')
    
    # Multi-stock parameters
    parser.add_argument('--multi_stock', action='store_true', help='Train on multiple stocks')
    parser.add_argument('--episodes_per_stock', type=int, default=5, help='Episodes per stock before rotation')
    parser.add_argument('--max_stocks', type=int, default=20, help='Maximum number of stocks to train on')
    parser.add_argument('--min_training_days', type=int, default=500, help='Minimum training days per stock')
    
    # Data split parameters
    parser.add_argument('--train_cutoff_date', type=str, default='2022-12-31', help='Training data cutoff date')
    parser.add_argument('--eval_start_date', type=str, default='2023-01-01', help='Evaluation data start date')
    
    # Control parameters
    parser.add_argument('--save_every', type=int, default=20, help='Save model every N episodes')
    parser.add_argument('--eval_every', type=int, default=100, help='Evaluate every N episodes')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model instead of training')
    parser.add_argument('--rl_model', type=str, help='RL model checkpoint for evaluation')
    parser.add_argument('--debug', action='store_true', help='Enable debug output during training')
    
    # Filtering parameters
    parser.add_argument('--exclude_tickers', type=str, nargs='*', default=['^TNX', '^VIX'], 
                       help='Ticker patterns to exclude')
    parser.add_argument('--include_only', type=str, nargs='*', help='Only include these tickers')
    
    args = parser.parse_args()
    
    # Setup paths
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_dir, args.data_dir)
    
    print(f"🚀 Vanilla RL Training")
    print(f"📂 Data directory: {data_path}")
    print(f"💾 Output directory: {args.out_dir}")
    print(f"📅 Train cutoff: {args.train_cutoff_date}")
    print(f"🧪 Eval start: {args.eval_start_date}")
    print(f"🎯 Multi-stock mode: {args.multi_stock}")
    print(f"🐛 Debug mode: {args.debug}")
    
    if args.evaluate:
        # Evaluate mode
        if args.rl_model:
            model = load_rl_model(args.device, args.out_dir, args.rl_model)
        else:
            model = load_model(args.device, args.out_dir, args.model_file)
        
        if model:
            if args.multi_stock:
                # Multi-stock evaluation
                trainer = MultiStockRLTrainer(model, args.device)
                stocks = trainer.get_available_stocks(data_path, args.exclude_tickers)
                if args.include_only:
                    stocks = [s for s in stocks if s in args.include_only]
                
                results = trainer.evaluate_on_multiple_stocks(
                    stocks[:10],  # Evaluate on first 10 stocks
                    data_path,
                    args.train_cutoff_date,
                    args.eval_start_date,
                    debug=args.debug
                )
                
                # Save results
                results_file = os.path.join(args.out_dir, f'rl_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Evaluation results saved to {results_file}")
            else:
                # Single stock evaluation
                metrics = evaluate_rl_model(model, args.ticker, data_path, args.device,
                                          cutoff_date=args.train_cutoff_date,
                                          debug=args.debug)
                print(f"🎯 Final evaluation: {metrics}")
    else:
        # Training mode
        print(f"🎯 Training mode: {args.episodes} episodes")
        
        # Load base model
        model = load_model(args.device, args.out_dir, args.model_file)
        if model is None:
            print("❌ Failed to load base model")
            return
        
        if args.multi_stock:
            # Multi-stock training
            trainer = MultiStockRLTrainer(
                model=model,
                device=args.device,
                learning_rate=args.lr,
                gamma=args.gamma,
                entropy_coeff=args.entropy_coeff
            )
            
            # Get available stocks
            stocks = trainer.get_available_stocks(data_path, args.exclude_tickers)
            if args.include_only:
                stocks = [s for s in stocks if s in args.include_only]
            
            # Validate stock data
            valid_stocks = trainer.validate_stock_data(
                stocks, data_path, args.train_cutoff_date, args.min_training_days
            )
            
            if len(valid_stocks) == 0:
                print("❌ No valid stocks found for training")
                return
            
            # Limit number of stocks if specified
            if args.max_stocks > 0:
                valid_stocks = valid_stocks[:args.max_stocks]
            
            print(f"🎯 Training on {len(valid_stocks)} stocks")
            
            # Train
            trainer.train_multi_stock(
                stocks=valid_stocks,
                data_dir=data_path,
                num_episodes=args.episodes,
                episodes_per_stock=args.episodes_per_stock,
                save_every=args.save_every,
                eval_every=args.eval_every,
                out_dir=args.out_dir,
                train_cutoff_date=args.train_cutoff_date,
                temperature=args.temperature,
                debug=args.debug
            )
            
            # Final evaluation on subset of stocks
            print(f"\n🧪 Final evaluation...")
            eval_stocks = valid_stocks[:min(10, len(valid_stocks))]
            final_results = trainer.evaluate_on_multiple_stocks(
                eval_stocks, data_path, args.train_cutoff_date, args.eval_start_date, debug=args.debug
            )
            
            # Save final results
            results_file = os.path.join(args.out_dir, f'rl_final_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            print(f"💾 Final results saved to {results_file}")
        else:
            # Single stock training
            trainer = RLTrainer(
                model=model,
                device=args.device,
                learning_rate=args.lr,
                gamma=args.gamma,
                entropy_coeff=args.entropy_coeff
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