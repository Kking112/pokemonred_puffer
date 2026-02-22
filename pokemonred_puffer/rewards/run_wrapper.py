"""
Run Penalty Reward Wrapper

This module provides a reward environment that penalizes running from battles
and rewards winning NPC trainer battles. This helps agents learn to fight
instead of repeatedly trying to run from trainer battles (which always fails).

Key Memory Addresses:
- wIsInBattle: 0=overworld, 1=wild encounter, 2=trainer battle
- wBattleResult: Battle outcome after fight ends
"""

import numpy as np
from omegaconf import DictConfig

from pokemonred_puffer.rewards.baseline import ObjectRewardRequiredEventsMapIds


class RunPenaltyRewardEnv(ObjectRewardRequiredEventsMapIds):
    """
    Reward environment that:
    1. Penalizes run attempts (especially useless in trainer battles)
    2. Rewards winning trainer battles
    
    This encourages the agent to fight battles rather than trying to escape.
    """

    def __init__(self, env_config: DictConfig, reward_config: DictConfig):
        super().__init__(env_config, reward_config)
        # Track battle state between steps
        self.prev_battle_type = 0  # 0=overworld, 1=wild, 2=trainer
        self.trainer_wins = 0
        self.run_attempts = 0
        self.prev_run_attempt_count = 0

    def get_game_state_reward(self) -> dict[str, float]:
        # Get the base rewards from parent class
        rewards = super().get_game_state_reward()

        # Current battle state
        current_battle_type = self.read_m("wIsInBattle")
        
        # Detect run attempts by checking wPlayerMoveNum when in battle
        # In Pokemon Red, selecting "RUN" sets certain battle state flags
        # We can also detect run attempts via the wNumRunAttempts counter
        run_attempt_addr = self.pyboy.symbol_lookup("wNumRunAttempts")[1]
        current_run_attempts = self.pyboy.memory[run_attempt_addr]
        
        # Count new run attempts this step
        new_run_attempts = 0
        if current_run_attempts > self.prev_run_attempt_count:
            new_run_attempts = current_run_attempts - self.prev_run_attempt_count
            self.run_attempts += new_run_attempts
        
        # Reset run attempt counter when not in battle (it resets in game too)
        if current_battle_type == 0:
            self.prev_run_attempt_count = 0
        else:
            self.prev_run_attempt_count = current_run_attempts
        
        # Detect trainer battle wins
        # A win is when we were in trainer battle (2) and now we're not (0)
        # and wBattleResult indicates a win (0 = win, 1 = lose, 2 = run)
        trainer_win_this_step = 0
        if self.prev_battle_type == 2 and current_battle_type == 0:
            battle_result = self.read_m("wBattleResult")
            if battle_result == 0:  # 0 = win
                self.trainer_wins += 1
                trainer_win_this_step = 1
        
        # Update battle type for next step
        self.prev_battle_type = current_battle_type
        
        # Add run penalty (negative reward per run attempt)
        # This is especially important in trainer battles where running always fails
        run_penalty = self.reward_config.get("run_penalty", -2.0)
        rewards["run_penalty"] = run_penalty * new_run_attempts
        
        # Add trainer battle win reward
        trainer_win_reward = self.reward_config.get("trainer_battle_win", 5.0)
        rewards["trainer_battle_win"] = trainer_win_reward * trainer_win_this_step
        
        # Also track cumulative stats for logging
        rewards["run_attempts_total"] = self.run_attempts * 0.0  # 0 weight, just for tracking
        rewards["trainer_wins_total"] = self.trainer_wins * 0.0  # 0 weight, just for tracking
        
        return rewards

    def reset(self, seed=None, options=None):
        """Reset the tracking variables on environment reset."""
        self.prev_battle_type = 0
        self.prev_run_attempt_count = 0
        # Don't reset trainer_wins and run_attempts - they're cumulative for logging
        return super().reset(seed=seed, options=options)
