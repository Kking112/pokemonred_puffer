"""
Main training loop for DreamerV3 on Pokemon Red.

This module integrates all DreamerV3 components and handles the complete
training procedure including:
- Experience collection from environments
- World model training
- Actor-critic training via imagination
- Logging and checkpointing
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.models.text_encoder import TransformerTextEncoder
from pokemonred_puffer.models.dreamer.world_model import DreamerV3WorldModel
from pokemonred_puffer.models.dreamer.actor_critic import (
    DreamerV3Actor,
    DreamerV3CriticEnsemble,
    ActorCriticTrainer,
)
from pokemonred_puffer.models.dreamer.world_model_trainer import (
    WorldModelTrainer,
    ImaginationRollout,
)
from pokemonred_puffer.models.dreamer.replay_buffer import MultiEnvReplayBuffer
from pokemonred_puffer.models.dreamer.utils import symlog, symexp


class DreamerV3Trainer:
    """
    Main trainer for DreamerV3 on Pokemon Red.
    """
    
    def __init__(
        self,
        config: DictConfig,
        env_creator,
        device: str = 'cuda',
        use_wandb: bool = False,
    ):
        """
        Initialize DreamerV3 trainer.
        
        Args:
            config: Configuration dictionary
            env_creator: Function to create environments
            device: Device to run on
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Create environments (we'll use a single env for now, can be extended)
        self.env = env_creator()
        
        # Get observation and action spaces
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        
        # Extract observation shapes and dtypes
        self.observation_shapes = {
            name: space.shape for name, space in obs_space.spaces.items()
        }
        self.observation_dtypes = {
            name: space.dtype for name, space in obs_space.spaces.items()
        }
        
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        self.discrete_actions = hasattr(action_space, 'n')
        
        # Build models
        self._build_models()
        
        # Create replay buffer
        self.replay_buffer = MultiEnvReplayBuffer(
            capacity=config.dreamer.replay_capacity,
            observation_shapes=self.observation_shapes,
            observation_dtypes=self.observation_dtypes,
            action_dim=self.action_dim,
            num_envs=config.dreamer.num_envs,
            sequence_length=config.dreamer.sequence_length,
            batch_size=config.dreamer.batch_size,
        )
        
        # Create trainers
        self.world_model_trainer = WorldModelTrainer(
            world_model=self.world_model,
            learning_rate=config.dreamer.world_model_lr,
            grad_clip=config.dreamer.grad_clip,
            kl_weight=config.dreamer.kl_weight,
            use_amp=config.dreamer.use_amp,
        )
        
        self.actor_critic_trainer = ActorCriticTrainer(
            actor=self.actor,
            critic=self.critic,
            actor_lr=config.dreamer.actor_lr,
            critic_lr=config.dreamer.critic_lr,
            discount=config.dreamer.discount,
            lambda_=config.dreamer.lambda_,
            entropy_coef=config.dreamer.entropy_coef,
        )
        
        # Create imagination rollout helper
        self.imagination = ImaginationRollout(
            world_model=self.world_model,
            actor=self.actor,
            horizon=config.dreamer.imagination_horizon,
        )
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # Metrics storage
        self.metrics = {}
        
    def _build_models(self):
        """Build all models."""
        config = self.config.dreamer
        
        # Text encoder
        self.text_encoder = TransformerTextEncoder(
            vocab_size=256,
            embed_dim=config.text_embed_dim,
            num_heads=config.text_num_heads,
            num_layers=config.text_num_layers,
            output_dim=config.text_output_dim,
        ).to(self.device)
        
        # Calculate state dimension (non-image observations)
        # This is a simplified calculation - adjust based on actual observations
        state_dim = sum([
            np.prod(shape) for name, shape in self.observation_shapes.items()
            if name not in ['screen', 'visited_mask', 'global_map', 'text']
        ])
        
        # World model
        screen_shape = self.observation_shapes['screen']
        self.world_model = DreamerV3WorldModel(
            action_dim=self.action_dim,
            screen_shape=screen_shape,
            text_encoder=self.text_encoder,
            state_dim=int(state_dim),
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            stoch_classes=config.stoch_classes,
            hidden_dim=config.hidden_dim,
            encoder_dim=config.encoder_dim,
        ).to(self.device)
        
        # Actor
        latent_dim = config.deter_dim + config.stoch_dim * config.stoch_classes
        self.actor = DreamerV3Actor(
            latent_dim=latent_dim,
            action_dim=self.action_dim,
            hidden_dims=[config.actor_hidden_dim] * config.actor_num_layers,
            discrete=self.discrete_actions,
        ).to(self.device)
        
        # Critic ensemble
        self.critic = DreamerV3CriticEnsemble(
            latent_dim=latent_dim,
            num_critics=config.num_critics,
            hidden_dims=[config.critic_hidden_dim] * config.critic_num_layers,
        ).to(self.device)
    
    def collect_experience(self, num_steps: int) -> Dict[str, float]:
        """
        Collect experience from the environment.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary of collection metrics
        """
        obs = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
        
        total_reward = 0.0
        episodes_finished = 0
        
        for _ in range(num_steps):
            # Convert observation to tensor
            obs_tensor = {}
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    obs_tensor[key] = torch.from_numpy(val).unsqueeze(0).to(self.device)
                else:
                    obs_tensor[key] = torch.tensor([val]).to(self.device)
            
            # Get action from actor (exploration mode)
            with torch.no_grad():
                # Encode observation
                obs_embedding = self.world_model.encoder(obs_tensor)
                
                # If we don't have a state yet, initialize
                if not hasattr(self, 'current_state'):
                    self.current_state = self.world_model.rssm.initial_state(1, self.device)
                
                # Update state (using posterior during collection)
                # Create dummy action for first step
                if not hasattr(self, 'last_action'):
                    self.last_action = torch.zeros(1, self.action_dim).to(self.device)
                
                self.current_state, _, _ = self.world_model.rssm.observe_step(
                    self.current_state,
                    self.last_action,
                    obs_embedding,
                )
                
                # Get latent and sample action
                latent = self.world_model.rssm.get_latent(self.current_state)
                action, _ = self.actor(latent, deterministic=False)
                
                self.last_action = action
            
            # Execute action in environment
            action_np = action.cpu().numpy().squeeze()
            if self.discrete_actions:
                action_np = int(action_np)
            
            result = self.env.step(action_np)
            next_obs, reward, done, truncated, info = result if len(result) == 5 else (*result, {})
            
            # Add to replay buffer
            self.replay_buffer.add(
                env_id=0,  # Single env for now
                observation=obs,
                action=action.cpu().numpy().squeeze(),
                reward=reward,
                done=done or truncated,
            )
            
            # Update metrics
            total_reward += reward
            self.episode_reward += reward
            self.episode_length += 1
            self.global_step += 1
            
            # Handle episode end
            if done or truncated:
                episodes_finished += 1
                self.episode_count += 1
                
                # Reset state
                if hasattr(self, 'current_state'):
                    delattr(self, 'current_state')
                if hasattr(self, 'last_action'):
                    delattr(self, 'last_action')
                
                # Log episode metrics
                if self.use_wandb:
                    wandb.log({
                        'episode/reward': self.episode_reward,
                        'episode/length': self.episode_length,
                        'episode/count': self.episode_count,
                    }, step=self.global_step)
                
                self.episode_reward = 0.0
                self.episode_length = 0
                
                obs = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
            else:
                obs = next_obs
        
        return {
            'collection/total_reward': total_reward,
            'collection/episodes_finished': episodes_finished,
        }
    
    def train_world_model(self, num_steps: int) -> Dict[str, float]:
        """
        Train the world model.
        
        Args:
            num_steps: Number of training steps
            
        Returns:
            Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready():
            return {'world_model/skipped': 1.0}
        
        metrics = {}
        
        for _ in range(num_steps):
            # Sample batch
            batch = self.replay_buffer.sample_sequences(device=self.device)
            
            # Train step
            step_metrics = self.world_model_trainer.train_step(batch)
            
            # Accumulate metrics
            for key, val in step_metrics.items():
                if key not in metrics:
                    metrics[key] = 0.0
                metrics[key] += val / num_steps
        
        return {f'world_model/{k}': v for k, v in metrics.items()}
    
    def train_actor_critic(self, num_steps: int) -> Dict[str, float]:
        """
        Train actor and critic via imagination.
        
        Args:
            num_steps: Number of training steps
            
        Returns:
            Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready():
            return {'actor_critic/skipped': 1.0}
        
        metrics = {}
        
        for _ in range(num_steps):
            # Sample batch to get initial states
            batch = self.replay_buffer.sample_sequences(device=self.device)
            
            # Get initial latent states from world model
            with torch.no_grad():
                observations = batch['observations']
                actions = batch['actions']
                
                # Encode first observation
                batch_size = actions.shape[0]
                first_obs = {key: val[:, 0] for key, val in observations.items()}
                obs_embedding = self.world_model.encoder(first_obs)
                
                # Initialize states
                initial_states = self.world_model.rssm.initial_state(batch_size, self.device)
                
                # Get posterior state for first timestep
                first_action = actions[:, 0]
                initial_states, _, _ = self.world_model.rssm.observe_step(
                    initial_states,
                    first_action,
                    obs_embedding,
                )
            
            # Imagination rollout
            with torch.no_grad():
                imagination_data = self.imagination.rollout(
                    initial_states=initial_states,
                    deterministic=False,
                )
            
            # Train actor and critic
            step_metrics = self.actor_critic_trainer.train_step(
                latents=imagination_data['latents'],
                rewards=imagination_data['rewards'],
                continues=imagination_data['continues'],
            )
            
            # Accumulate metrics
            for key, val in step_metrics.items():
                if key not in metrics:
                    metrics[key] = 0.0
                metrics[key] += val / num_steps
        
        return {f'actor_critic/{k}': v for k, v in metrics.items()}
    
    def train(self):
        """Main training loop."""
        config = self.config.dreamer
        
        print(f"Starting DreamerV3 training on device: {self.device}")
        print(f"Replay buffer capacity: {self.replay_buffer.total_capacity()}")
        print(f"Prefilling replay buffer with {config.prefill_steps} steps...")
        
        # Prefill replay buffer
        self.collect_experience(config.prefill_steps)
        
        print(f"Replay buffer size: {len(self.replay_buffer)}")
        print("Starting training loop...")
        
        # Main training loop
        for iteration in range(config.num_iterations):
            iter_start_time = time.time()
            
            # Collect experience
            collection_metrics = self.collect_experience(config.collect_interval)
            
            # Train world model
            wm_metrics = self.train_world_model(config.world_model_train_steps)
            
            # Train actor-critic
            ac_metrics = self.train_actor_critic(config.actor_critic_train_steps)
            
            # Combine metrics
            all_metrics = {
                **collection_metrics,
                **wm_metrics,
                **ac_metrics,
                'training/iteration': iteration,
                'training/global_step': self.global_step,
                'training/replay_buffer_size': len(self.replay_buffer),
                'training/iteration_time': time.time() - iter_start_time,
            }
            
            # Log metrics
            if self.use_wandb:
                wandb.log(all_metrics, step=self.global_step)
            
            # Print progress
            if iteration % config.log_interval == 0:
                print(f"\n[Iteration {iteration}]")
                print(f"  Global step: {self.global_step}")
                print(f"  Replay buffer: {len(self.replay_buffer)}/{self.replay_buffer.total_capacity()}")
                if 'world_model/total_loss' in all_metrics:
                    print(f"  WM Loss: {all_metrics['world_model/total_loss']:.4f}")
                if 'actor_critic/actor_loss' in all_metrics:
                    print(f"  Actor Loss: {all_metrics['actor_critic/actor_loss']:.4f}")
                if 'actor_critic/critic_loss' in all_metrics:
                    print(f"  Critic Loss: {all_metrics['actor_critic/critic_loss']:.4f}")
                print(f"  Collection reward: {collection_metrics.get('collection/total_reward', 0.0):.2f}")
            
            # Save checkpoint
            if iteration % config.checkpoint_interval == 0 and iteration > 0:
                self.save_checkpoint(iteration)
        
        print("\nTraining complete!")
    
    def save_checkpoint(self, iteration: int):
        """
        Save training checkpoint.
        
        Args:
            iteration: Current training iteration
        """
        checkpoint_dir = Path(self.config.dreamer.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.pt"
        
        checkpoint = {
            'iteration': iteration,
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'world_model_optimizer': self.world_model_trainer.optimizer.state_dict(),
            'actor_optimizer': self.actor_critic_trainer.actor_optimizer.state_dict(),
            'critic_optimizer': self.actor_critic_trainer.critic_optimizer.state_dict(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        
        self.world_model_trainer.optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_critic_trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.actor_critic_trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Iteration: {checkpoint['iteration']}")
        print(f"  Global step: {self.global_step}")
        print(f"  Episodes: {self.episode_count}")


def main():
    """Main entry point for DreamerV3 training."""
    parser = argparse.ArgumentParser(description='Train DreamerV3 on Pokemon Red')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=OmegaConf.to_container(config, resolve=True),
            name=f"dreamerv3_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    
    # Create environment creator
    # This is a simplified version - adapt based on your setup
    def env_creator():
        from pokemonred_puffer.environment import RedGymEnv
        return RedGymEnv(config.env)
    
    # Create trainer
    trainer = DreamerV3Trainer(
        config=config,
        env_creator=env_creator,
        device=args.device,
        use_wandb=args.wandb,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(trainer.global_step // config.dreamer.collect_interval)
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

