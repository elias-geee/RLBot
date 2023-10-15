import numpy as np
from enum import Enum
import rlgym
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, AlignBallGoal
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward, FaceBallReward, TouchBallReward, LiuDistancePlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward, LiuDistanceBallToGoalReward
#from rlgym.utils.reward_functions.common_rewards.tryout import Reward_Values
from rlgym.utils.reward_functions import CombinedReward
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from rlgym.utils.gamestates import GameState, PlayerData
#from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward
#from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED
#from rlgym.utils import RewardFunction, math
#from rlgym.rocket_learn.utils.gamestate_encoding import StateConstants


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = 0
        self.logger.record("BallTouches", value)
        return True


if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    agents_per_match = 2
    self_play=True
    num_instances = 2 #number of rl windows
    target_steps = 100_000
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps

    print(f"fps={fps}, gamma={gamma})")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,  # 3v3 to get as many agents going as possible, will make results more noisy
            tick_skip=frame_skip,
            reward_function=CombinedReward(
            (
                    VelocityPlayerToBallReward(),
                    VelocityBallToGoalReward(),
                    EventReward(
                        team_goal= 100.0,
                        concede = -100.0,
                        shot = 5.0,
                        save = 30.0,
                        demo = 10.0,
                    ),
                    FaceBallReward(),
                    TouchBallReward(),
                    LiuDistancePlayerToBallReward(),
                    LiuDistanceBallToGoalReward(),
                    AlignBallGoal(),
            ),
            (0.5,0.9,1.0,0.008,0.15,0.08,0.05,0.004)),  # Simple reward since example code #V_2 added FaceBallReward, ToucBallReward, LiuDistancePlayertoBall, LiuDistanceBalltoGoal, AlignGoal
            spawn_opponents=True,
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)            # Start num_instances instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try:
        model = PPO.load(
        "models/exit_save.zip",
        env,
        #custom_objects=dict(n_envs=env.num_envs, _last_obs=None),  # Need this to change number of agents
        device="auto",  # Need to set device again (if using a specific one)
        #force_reset=True  # Make SB3 reset the env so it doesn't think we're continuing from last state
    )

    except:
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn = Tanh,
            net_arch =[512,512, dict(pi=[256,256,256], vf=[256,256,256])],
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=1,  # PPO calls for multiple epochs
            policy_kwargs = policy_kwargs,
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,             # Batch size as high as possible within reason
            n_steps=steps,                # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")
    all_callbacks = [callback, TensorboardCallback()]



    while True:

        model.learn(25_000_000, callback=all_callbacks, reset_num_timesteps=False)
        model.save("models/exit_save")
        model.save("mmr_models/"+f"{model.num_timesteps}")



    # Now, if one wants to load a trained model from a checkpoint, use this function
    # This will contain all the attributes of the original model
    # Any attribute can be overwritten by using the custom_objects parameter,
    # which includes n_envs (number of agents), which has to be overwritten to use a different amount

    # Use reset_num_timesteps=False to keep going with same logger/checkpoints
    #model.learn(100_000_000, callback=callback, reset_num_timesteps=False)
