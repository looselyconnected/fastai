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
        
        # Determine directions (positive = up, negative = down)
        pred_direction = 1 if pred_cumulative > 0 else -1
        actual_direction = 1 if actual_cumulative > 0 else -1
        
        # Reward is 1 if directions match, -1 if they don't
        reward = 1.0 if pred_direction == actual_direction else -1.0
        
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
        
        for step in range(max_new_tokens):
            # Ensure we don't exceed block size
            if len(current_sequence) >= self.model.config.block_size:
                # Crop the sequence to fit in block size
                current_sequence = current_sequence[-self.model.config.block_size + 1:]
            
            # Get model predictions (need gradients for policy gradient)
            logits, _ = self.model(current_sequence.unsqueeze(0))
            logits = logits[0, -1, :] / temperature  # Get last token logits
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            sampled_token = torch.multinomial(probs, 1)
            
            # Store log probability
            log_prob = F.log_softmax(logits, dim=-1)[sampled_token]
            log_probs.append(log_prob)
            
            # Append to sequence (detach to avoid accumulating gradients through the sequence)
            current_sequence = torch.cat([current_sequence, sampled_token.detach()])
        
        # Extract the generated portion
        generated_tokens = current_sequence[len(context_tokens):]
        
        # Compute reward every day (every 9 tokens)
        for day in range(predict_days):
            start_idx = day * tokens_per_day
            end_idx = (day + 1) * tokens_per_day
            
            if end_idx <= len(target_tokens):
                # Get tokens for this day
                pred_day_tokens = generated_tokens[start_idx:end_idx]
                actual_day_tokens = target_tokens[start_idx:end_idx]
                
                # Compute reward for this day
                day_reward = self.compute_20_day_reward(pred_day_tokens, actual_day_tokens, debug=debug and day == 0)
                rewards.append(day_reward)
            else:
                # If we don't have ground truth, give neutral reward
                rewards.append(0.0)
        
        if debug:
            print(f"📊 Episode stats: {len(rewards)} day rewards, avg reward: {np.mean(rewards):.3f}")
        
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
        
        # Compute discounted rewards
        discounted_rewards = self.compute_discounted_rewards(rewards)
        
        # Update baseline (simple exponential moving average)
        episode_return = sum(rewards)
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * episode_return
        
        # Compute policy gradient loss with entropy regularization
        policy_loss = 0.0
        entropy_loss = 0.0
        
        for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
            # Subtract baseline to reduce variance
            advantage = discounted_reward - self.baseline
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
        self.episode_rewards.append(episode_return)
        self.episode_losses.append(total_loss.item())
        
        # Compute direction accuracy
        direction_correct = sum(1 for r in rewards if r > 0) / len(rewards)
        self.prediction_accuracy.append(direction_correct)
        
        stats = {
            'episode_return': episode_return,
            'policy_loss': policy_loss.item(),
            'total_loss': total_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'direction_accuracy': direction_correct,
            'baseline': self.baseline,
            'avg_reward': np.mean(rewards)
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
            
            # Randomly sample a training window
            max_start_idx = len(all_data_df) - min_context_days - predict_days
            if max_start_idx <= 0:
                print("❌ Not enough data for training")
                break
            
            start_idx = random.randint(0, max_start_idx)
            context_end_idx = start_idx + min_context_days
            target_end_idx = context_end_idx + predict_days
            
            # Get context and target data
            context_df = all_data_df.iloc[start_idx:context_end_idx]
            target_df = all_data_df.iloc[context_end_idx:target_end_idx]
            
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
    
    metrics = {
        'direction_reward': reward,
        'predicted_cumulative': pred_cumulative,
        'actual_cumulative': actual_cumulative,
        'direction_match': reward > 0,
        'mae': np.mean(np.abs(pred_std_values - actual_std_values)),
        'mse': np.mean((pred_std_values - actual_std_values)**2)
    }
    
    if debug:
        print(f"\n📊 Evaluation Results:")
        print(f"  Direction reward: {metrics['direction_reward']}")
        print(f"  Predicted cumulative: {metrics['predicted_cumulative']:.3f}")
        print(f"  Actual cumulative: {metrics['actual_cumulative']:.3f}")
        print(f"  Direction match: {metrics['direction_match']}")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  MSE: {metrics['mse']:.3f}")
    
    return metrics


def main():
    """Main function for RL training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RL training for stock prediction')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--model_file', type=str, default='ckpt.pt', help='Base model checkpoint')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='mps', help='Device (cpu/cuda/mps)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model instead of training')
    parser.add_argument('--rl_model', type=str, help='RL model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_dir, args.data_dir)
    
    if args.evaluate:
        # Evaluate mode
        if args.rl_model:
            model = load_rl_model(args.device, args.out_dir, args.rl_model)
        else:
            model = load_model(args.device, args.out_dir, args.model_file)
        
        if model:
            metrics = evaluate_rl_model(model, args.ticker, data_path, args.device)
            print(f"🎯 Final evaluation: {metrics}")
    else:
        # Training mode
        print(f"🚀 Starting RL training with {args.episodes} episodes")
        
        # Load base model
        model = load_model(args.device, args.out_dir, args.model_file)
        
        if model is None:
            print("❌ Failed to load base model")
            return
        
        # Create RL trainer
        trainer = RLTrainer(model, args.device, args.lr)
        
        # Train
        trainer.train(
            ticker=args.ticker,
            data_dir=data_path,
            num_episodes=args.episodes,
            temperature=args.temperature,
            out_dir=args.out_dir
        )


if __name__ == "__main__":
    main()