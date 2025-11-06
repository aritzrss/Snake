import gymnasium as gym
import snake_env
from stable_baselines3 import PPO
import time
import numpy as np

def visualize_agent(model_path, num_episodes=5, delay=0.1):
    """
    Visualize the trained agent playing Snake
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to play
        delay: Delay between steps (seconds) for visualization
    """
    print("="*60)
    print("VISUALIZING TRAINED SNAKE AGENT")
    print("="*60)
    
    # Load the trained model
    print(f"\nLoading model from: {model_path}")
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've trained a model first!")
        return
    
    # Create environment with rendering
    env = gym.make("Snake-v0", render_mode="human")
    
    print(f"\nPlaying {num_episodes} episodes...\n")
    
    episode_rewards = []
    episode_lengths = []
    max_snake_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        max_length = 3  # Initial snake length
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not (done or truncated):
            # Get action from the trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Track max snake length (if available in info)
            if 'snake_length' in info:
                max_length = max(max_length, info['snake_length'])
            
            # Render and add delay for visualization
            env.render()
            time.sleep(delay)
            
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        max_snake_lengths.append(max_length)
        
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Max Snake Length: {max_length}")
        print()
    
    env.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average Max Snake Length: {np.mean(max_snake_lengths):.2f} ± {np.std(max_snake_lengths):.2f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Best Snake Length: {np.max(max_snake_lengths)}")
    print("="*60)


def compare_models(model_paths, labels, num_episodes=3):
    """
    Compare multiple models side by side
    
    Args:
        model_paths: List of paths to saved models
        labels: List of labels for each model
        num_episodes: Number of episodes per model
    """
    print("="*60)
    print("COMPARING MODELS")
    print("="*60)
    
    results = {}
    
    for model_path, label in zip(model_paths, labels):
        print(f"\nTesting: {label}")
        print("-"*60)
        
        try:
            model = PPO.load(model_path)
            env = gym.make("Snake-v0", render_mode=None)  # No rendering for comparison
            
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False
                episode_reward = 0
                steps = 0
                
                while not (done or truncated):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
            
            env.close()
            
            results[label] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths)
            }
            
            print(f"Mean Reward: {results[label]['mean_reward']:.2f} ± {results[label]['std_reward']:.2f}")
            print(f"Mean Length: {results[label]['mean_length']:.2f} ± {results[label]['std_length']:.2f}")
            
        except Exception as e:
            print(f"Error loading model {label}: {e}")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for label, stats in results.items():
        print(f"\n{label}:")
        print(f"  Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize trained Snake agent')
    parser.add_argument('--model', type=str, default='models/snake_curriculum/best_model',
                        help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between steps (seconds)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare best and final models')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare different checkpoints
        model_paths = [
            'models/snake_curriculum/best_model',
            'models/snake_curriculum/final_model'
        ]
        labels = ['Best Model', 'Final Model']
        compare_models(model_paths, labels, num_episodes=5)
    else:
        # Visualize single model
        visualize_agent(args.model, args.episodes, args.delay)