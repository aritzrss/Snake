import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class CurriculumWrapper(Wrapper):
    """
    Wrapper that handles curriculum learning by updating environment parameters
    on each reset without recreating the environment (which causes issues with PPO).
    """
    
    def __init__(self, env, curriculum_stages):
        """
        Args:
            env: The base Snake environment
            curriculum_stages: List of stage configurations
        """
        super().__init__(env)
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        
    def set_curriculum_stage(self, stage_index):
        """Update the current curriculum stage"""
        if stage_index < len(self.curriculum_stages):
            self.current_stage = stage_index
            print(f"Curriculum wrapper: Now at stage {stage_index + 1}")
        
    def reset(self, **kwargs):
        """Reset the environment with current curriculum stage parameters"""
        # Get current stage configuration
        stage_config = self.curriculum_stages[self.current_stage]
        
        # Update environment parameters
        self.env.tableSize = stage_config['grid_size']
        self.env.tableSizeObs = stage_config['grid_size']
        self.env.halfTable = int(self.env.tableSize / 2)
        self.env.initial_snake_length = stage_config['initial_snake_length']
        
        # Recreate the image buffer with the new size
        self.env.img = np.zeros((self.env.tableSize, self.env.tableSize, 3), dtype='uint8')
        
        # Call the base environment's reset
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Pass through to the base environment"""
        return self.env.step(action)