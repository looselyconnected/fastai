"""
Multi-Stock PPO RL Training Script

This script performs PPO reinforcement learning fine-tuning across all available stock data.
It loads all TICKER_train.csv files from the data directory and trains the model on multiple
stocks simultaneously for better generalization and robustness.

Key features:
- Multi-stock training for better generalization
- Proper train/test split (2022 for training, 2023+ for testing)
- Curriculum learning with stock rotation
- Comprehensive evaluation across all stocks
- Model checkpointing and progress tracking
"""

import os
import glob
import torch
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Tuple
import random
import argparse
import json
from collections import defaultdict, deque
import logging

from model import GPT, load_model
from ppo_rl import PPOTrainer, load_ppo_model, evaluate_ppo_model
from rl import MultiStockRLTrainer, load_rl_model, evaluate_rl_model
from data import get_data_for_eval

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiStockPPOTrainer:
    """
    Multi-stock PPO trainer that handles training across multiple stocks.
    """
    
    def __init__(self, 
                 model: GPT,
                 device: str = 'cpu',
                 learning_rate: float = 1e-5,
                 value_lr: float = 1e-4,
                 **ppo_kwargs):
        """
        Initialize multi-stock PPO trainer.
        
        Args:
            model: Pre-trained GPT model
            device: Device to run on
            learning_rate: Policy learning rate
            value_lr: Value learning rate
            **ppo_kwargs: Additional PPO parameters
        """
        self.device = device
        self.model = model
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            value_lr=value_lr,
            **ppo_kwargs
        )
        
        # Track multi-stock statistics
        self.stock_performance = defaultdict(list)
        self.global_episode_count = 0
        self.stock_episode_counts = defaultdict(int)
        
        print(f"✓ Multi-Stock PPO Trainer initialized")
        print(f"  Policy LR: {learning_rate}, Value LR: {value_lr}")
    
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
        print(f"🚀 Starting Multi-Stock PPO Training")
        print(f"📚 {len(stocks)} stocks, {num_episodes} total episodes")
        print(f"🔄 {episodes_per_stock} episodes per stock rotation")
        print(f"📅 Training cutoff: {train_cutoff_date}")
        
        best_avg_accuracy = 0.0
        recent_rewards = deque(maxlen=20)
        stock_rotation_idx = 0
        
        for episode in range(num_episodes):
            # Update recovery schedule if in recovery mode
            self.ppo_trainer.update_recovery_schedule()
            
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
                min_context_days = 500
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
                
                # Train episode
                from data import encode_data
                context_tokens = encode_data(context_df).squeeze().to(self.device)
                target_tokens = encode_data(target_df).squeeze().to(self.device)
                
                episode_rewards, episode_stats = self.ppo_trainer.generate_episode(
                    context_tokens, target_tokens, temperature,
                    debug=(debug and episode % 50 == 0)
                )
                
                # Handle policy collapse in multi-stock training
                if episode_stats.get('collapse_detected', False):
                    print(f"💥 Multi-stock policy collapse detected at episode {episode} on {current_stock}")
                    print(f"   Prediction std: {episode_stats.get('prediction_std', 0):.6f}")
                    self.ppo_trainer.recover_from_collapse()
                
                # Update policy when buffer is full (skip if just recovered from collapse)
                if (self.ppo_trainer.buffer.size() >= self.ppo_trainer.batch_size and 
                    not episode_stats.get('collapse_detected', False)):
                    update_stats = self.ppo_trainer.update_policy(debug=(debug and episode % 50 == 0))
                
                # Track statistics
                self.ppo_trainer.episode_rewards.append(episode_stats['total_reward'])
                self.ppo_trainer.prediction_accuracy.append(episode_stats['direction_accuracy'])
                recent_rewards.append(episode_stats['total_reward'])
                
                # Track per-stock performance
                self.stock_performance[current_stock].append({
                    'episode': episode,
                    'reward': episode_stats['total_reward'],
                    'accuracy': episode_stats['direction_accuracy']
                })
                self.stock_episode_counts[current_stock] += 1
                
                # Print progress
                if debug and episode % 25 == 0:
                    recent_accuracy = np.mean([ep['accuracy'] for eps in self.stock_performance.values() 
                                             for ep in eps[-5:]])  # Last 5 episodes per stock
                    recent_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                    print(f"📈 Episode {episode}: {current_stock} acc={episode_stats['direction_accuracy']:.3f}, "
                          f"avg_acc={recent_accuracy:.3f}, avg_reward={recent_reward:.3f}")
                
            except Exception as e:
                print(f"❌ Error training on {current_stock} at episode {episode}: {e}")
                continue
            
            # Save model periodically
            if episode % save_every == 0 and episode > 0:
                avg_accuracy = np.mean([ep['accuracy'] for eps in self.stock_performance.values() 
                                      for ep in eps[-10:]])  # Last 10 episodes per stock
                if avg_accuracy > best_avg_accuracy:
                    best_avg_accuracy = avg_accuracy
                    filename = f'multi_stock_ppo_episode_{episode}_acc_{avg_accuracy:.3f}.pt'
                    self.ppo_trainer.save_model(out_dir, filename)
                    print(f"💾 Saved multi-stock model at episode {episode} with accuracy {avg_accuracy:.3f}")
            
            # Comprehensive evaluation
            if episode % eval_every == 0 and episode > 0:
                self.evaluate_on_multiple_stocks(stocks[:5], data_dir, train_cutoff_date, debug=True)
        
        # Save final model
        final_accuracy = np.mean([ep['accuracy'] for eps in self.stock_performance.values() 
                                for ep in eps[-20:]])  # Last 20 episodes per stock
        final_filename = f'multi_stock_ppo_final_acc_{final_accuracy:.3f}.pt'
        self.ppo_trainer.save_model(out_dir, final_filename)
        
        print(f"🎉 Multi-stock training complete! Final accuracy: {final_accuracy:.3f}")
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
                metrics = evaluate_ppo_model(
                    self.model, self.ppo_trainer.value_net, ticker, data_dir, self.device,
                    train_cutoff_date=train_cutoff_date,
                    eval_start_date=eval_start_date,
                    debug=debug
                )
                
                results[ticker] = metrics
                all_rewards.append(metrics.get('continuous_reward', 0))
                all_accuracies.append(1.0 if metrics.get('direction_match', False) else 0.0)
                
                if debug:
                    print(f"  {ticker}: reward={metrics.get('continuous_reward', 0):.3f}, "
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
        print("🎯 MULTI-STOCK PPO TRAINING SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_episodes = sum(len(eps) for eps in self.stock_performance.values())
        total_rewards = [ep['reward'] for eps in self.stock_performance.values() for ep in eps]
        total_accuracies = [ep['accuracy'] for eps in self.stock_performance.values() for ep in eps]
        
        print(f"Total episodes: {total_episodes}")
        print(f"Stocks trained: {len(self.stock_performance)}")
        print(f"Average reward: {np.mean(total_rewards):.3f}")
        print(f"Average accuracy: {np.mean(total_accuracies):.3f}")
        print(f"Best accuracy: {np.max(total_accuracies):.3f}")
        
        # Per-stock summary
        print(f"\nPer-stock performance:")
        for stock in sorted(self.stock_performance.keys()):
            episodes = self.stock_performance[stock]
            if episodes:
                avg_reward = np.mean([ep['reward'] for ep in episodes])
                avg_accuracy = np.mean([ep['accuracy'] for ep in episodes])
                best_accuracy = np.max([ep['accuracy'] for ep in episodes])
                print(f"  {stock:6}: {len(episodes):3} episodes, "
                      f"avg_reward={avg_reward:6.2f}, avg_acc={avg_accuracy:.3f}, best_acc={best_accuracy:.3f}")
        
        print("="*80)


def main():
    """Main function for multi-stock RL training."""
    parser = argparse.ArgumentParser(description='Multi-Stock RL Training')
    
    # Basic parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--model_file', type=str, default='ckpt.pt', help='Base model checkpoint')
    parser.add_argument('--device', type=str, default='mps', help='Device (cpu/cuda/mps)')
    
    # RL Mode Selection
    parser.add_argument('--use_ppo', action='store_true', help='Use PPO instead of vanilla RL')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=500, help='Total number of episodes')
    parser.add_argument('--episodes_per_stock', type=int, default=5, help='Episodes per stock before rotation')
    parser.add_argument('--lr', type=float, default=1e-5, help='Policy learning rate')
    parser.add_argument('--value_lr', type=float, default=1e-4, help='Value learning rate (PPO only)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    
    # PPO parameters (only used if --use_ppo is set)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (PPO only)')
    parser.add_argument('--epochs_per_update', type=int, default=2, help='Epochs per update (PPO only)')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon (PPO only)')
    
    # Vanilla RL parameters (only used if --use_ppo is not set)
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (vanilla RL only)')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Entropy coefficient (vanilla RL only)')
    
    # Data split parameters
    parser.add_argument('--train_cutoff_date', type=str, default='2022-12-31', help='Training data cutoff date')
    parser.add_argument('--eval_start_date', type=str, default='2023-01-01', help='Evaluation data start date')
    
    # Validation parameters
    parser.add_argument('--min_training_days', type=int, default=500, help='Minimum training days per stock')
    parser.add_argument('--max_stocks', type=int, default=50, help='Maximum number of stocks to train on')
    
    # Control parameters
    parser.add_argument('--save_every', type=int, default=50, help='Save model every N episodes')
    parser.add_argument('--eval_every', type=int, default=100, help='Evaluate every N episodes')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate existing model')
    parser.add_argument('--model_to_evaluate', type=str, help='Model file to evaluate')
    
    # Filtering parameters
    parser.add_argument('--exclude_tickers', type=str, nargs='*', default=['^TNX', '^VIX'], 
                       help='Ticker patterns to exclude')
    parser.add_argument('--include_only', type=str, nargs='*', help='Only include these tickers')
    
    args = parser.parse_args()
    
    # Setup paths
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_dir, args.data_dir)
    
    rl_mode = "PPO" if args.use_ppo else "Vanilla RL"
    print(f"🚀 Multi-Stock {rl_mode} Training")
    print(f"📂 Data directory: {data_path}")
    print(f"💾 Output directory: {args.out_dir}")
    print(f"📅 Train cutoff: {args.train_cutoff_date}")
    print(f"🧪 Eval start: {args.eval_start_date}")
    print(f"🎯 RL Mode: {rl_mode}")
    
    if args.evaluate_only:
        # Evaluation mode
        if not args.model_to_evaluate:
            print("❌ Please specify --model_to_evaluate for evaluation mode")
            return
        
        print(f"🧪 Evaluation mode: {args.model_to_evaluate}")
        
        # Load model based on RL mode
        if args.use_ppo:
            model, value_net = load_ppo_model(args.device, args.out_dir, args.model_to_evaluate)
            if not model or not value_net:
                print("❌ Failed to load PPO model")
                return
            
            # Create PPO trainer for evaluation
            trainer = MultiStockPPOTrainer(model, args.device)
            trainer.ppo_trainer.value_net = value_net
        else:
            model = load_rl_model(args.device, args.out_dir, args.model_to_evaluate)
            if not model:
                print("❌ Failed to load RL model")
                return
            
            # Create vanilla RL trainer for evaluation
            trainer = MultiStockRLTrainer(model, args.device)
        
        # Get stocks for evaluation
        stocks = trainer.get_available_stocks(data_path, args.exclude_tickers)
        if args.include_only:
            stocks = [s for s in stocks if s in args.include_only]
        
        # Evaluate
        results = trainer.evaluate_on_multiple_stocks(
            stocks[:10],  # Evaluate on first 10 stocks
            data_path,
            args.train_cutoff_date,
            args.eval_start_date,
            debug=True
        )
        
        # Save results
        results_file = os.path.join(args.out_dir, f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Evaluation results saved to {results_file}")
        
    else:
        # Training mode
        print(f"🎯 Training mode: {args.episodes} episodes")
        
        # Load base model
        model = load_model(args.device, args.out_dir, args.model_file)
        if model is None:
            print("❌ Failed to load base model")
            return
        
        # Create multi-stock trainer based on RL mode
        if args.use_ppo:
            trainer = MultiStockPPOTrainer(
                model=model,
                device=args.device,
                learning_rate=args.lr,
                value_lr=args.value_lr,
                batch_size=args.batch_size,
                epochs_per_update=args.epochs_per_update,
                clip_epsilon=args.clip_epsilon
            )
        else:
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
        
        # Train using appropriate method
        if args.use_ppo:
            trainer.train_multi_stock(
                stocks=valid_stocks,
                data_dir=data_path,
                num_episodes=args.episodes,
                episodes_per_stock=args.episodes_per_stock,
                save_every=args.save_every,
                eval_every=args.eval_every,
                out_dir=args.out_dir,
                train_cutoff_date=args.train_cutoff_date,
                temperature=args.temperature
            )
        else:
            trainer.train_multi_stock(
                stocks=valid_stocks,
                data_dir=data_path,
                num_episodes=args.episodes,
                episodes_per_stock=args.episodes_per_stock,
                save_every=args.save_every,
                eval_every=args.eval_every,
                out_dir=args.out_dir,
                train_cutoff_date=args.train_cutoff_date,
                temperature=args.temperature
            )
        
        # Final evaluation on subset of stocks
        print(f"\n🧪 Final evaluation...")
        eval_stocks = valid_stocks[:min(10, len(valid_stocks))]
        final_results = trainer.evaluate_on_multiple_stocks(
            eval_stocks, data_path, args.train_cutoff_date, args.eval_start_date, debug=True
        )
        
        # Save final results
        results_file = os.path.join(args.out_dir, f'final_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        print(f"💾 Final results saved to {results_file}")


if __name__ == "__main__":
    main()