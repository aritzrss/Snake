import gymnasium as gym
import snake_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime

class CurriculumCallback(BaseCallback):
    """
    Callback to manage curriculum learning progression
    """
    def __init__(self, curriculum_stages, stage_timesteps, verbose=0):
        super().__init__(verbose)
        self.curriculum_stages = curriculum_stages
        self.stage_timesteps = stage_timesteps
        self.current_stage = 0
        self.stage_start_timestep = 0
        
    def _on_step(self) -> bool:
        # Check if we should progress to next stage
        if self.current_stage < len(self.curriculum_stages) - 1:
            timesteps_in_stage = self.num_timesteps - self.stage_start_timestep
            if timesteps_in_stage >= self.stage_timesteps[self.current_stage]:
                self.current_stage += 1
                self.stage_start_timestep = self.num_timesteps
                print(f"\n{'='*60}")
                print(f"ADVANCING TO STAGE {self.current_stage + 1}/{len(self.curriculum_stages)}")
                print(f"Stage config: {self.curriculum_stages[self.current_stage]}")
                print(f"{'='*60}\n")
                
                # Update environment with new stage parameters
                # Note: This requires your Snake environment to support these parameters
                # You might need to recreate the environment here
                
        return True

class MetricsCallback(BaseCallback):
    """
    Callback to track and display training metrics
    """
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get episode rewards from the monitor
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                
                print(f"\nTimestep: {self.num_timesteps}")
                print(f"Mean Reward (last 100 episodes): {mean_reward:.2f}")
                print(f"Mean Episode Length: {mean_length:.2f}")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"New best mean reward: {self.best_mean_reward:.2f}! Saving model...")
                    self.model.save(f"{self.model_save_path}/best_model")
                    
        return True
    
    def _init_callback(self) -> None:
        # Create folder for saving models
        self.model_save_path = "models/snake_curriculum"
        os.makedirs(self.model_save_path, exist_ok=True)


def create_curriculum_stages():
    """
    Define curriculum learning stages
    Each stage makes the task progressively harder
    """
    stages = [
        {
            "name": "Stage 1: Small Grid - Learn Basics",
            "description": "Small environment to learn basic movement and apple collection",
            "config": {"grid_size": 10}  # Modify based on your env's parameters
        },
        {
            "name": "Stage 2: Medium Grid",
            "description": "Medium-sized grid to learn better navigation",
            "config": {"grid_size": 15}
        },
        {
            "name": "Stage 3: Larger Grid",
            "description": "Larger grid closer to final challenge",
            "config": {"grid_size": 18}
        },
        {
            "name": "Stage 4: Full Challenge",
            "description": "Full-size environment",
            "config": {"grid_size": 20}
        }
    ]
    
    return stages


def train_with_curriculum():
    """
    Main training function with curriculum learning
    """
    print("="*60)
    print("SNAKE CURRICULUM LEARNING WITH PPO")
    print("="*60)
    
    # Create curriculum stages
    curriculum_stages = create_curriculum_stages()
    
    # Timesteps per stage (adjust based on your needs)
    stage_timesteps = [50000, 100000, 150000, 200000]  # Total: 500k timesteps
    
    # Print curriculum plan
    print("\nCURRICULUM PLAN:")
    for i, stage in enumerate(curriculum_stages):
        print(f"\n{stage['name']}:")
        print(f"  Description: {stage['description']}")
        print(f"  Timesteps: {stage_timesteps[i]:,}")
        print(f"  Config: {stage['config']}")
    
    print("\n" + "="*60 + "\n")
    
    # Create initial environment (Stage 1)
    # Note: You may need to modify your Snake environment to accept these parameters
    env = gym.make("Snake-v0", render_mode=None)
    
    # Create model directory
    models_dir = "models/snake_curriculum"
    logs_dir = "logs/snake_curriculum"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        tensorboard_log=logs_dir
    )
    
    # Create callbacks
    metrics_callback = MetricsCallback(check_freq=5000)
    curriculum_callback = CurriculumCallback(curriculum_stages, stage_timesteps)
    
    # Train the model
    print("Starting training...\n")
    total_timesteps = sum(stage_timesteps)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[metrics_callback, curriculum_callback],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = f"{models_dir}/final_model"
        model.save(final_model_path)
        print(f"\nTraining complete! Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        model.save(f"{models_dir}/interrupted_model")
        print("Model saved!")
    
    env.close()
    return model


if __name__ == "__main__":
    # Run training
    model = train_with_curriculum()
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("Models saved in: ./models/snake_curriculum/")
    print("Logs saved in: ./logs/snake_curriculum/")
    print("="*60)