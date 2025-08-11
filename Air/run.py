import os
os.chdir('Air')
import torch
import datetime
from pathlib import Path
from gym.wrappers import FrameStack
import gymnasium as gym
import ale_py
from RL_brain import DQN,MetricLogger
import numpy as np
from env_preprocess import SkipFrame,GrayScaleObservation,ResizeObservation

gym.register_envs(ale_py)


use_cuda = torch.cuda.is_available()
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

env = gym.make('ALE/Asterix-v5',render_mode='rgb_array')
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)


RL = DQN(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)

seed = 64+50000
counter = 0
episodes = 100000
load = torch.load('checkpoints/mario_net_43.chkpt')
RL.net.load_state_dict(load['model'])
RL.exploration_rate = load['exploration_rate']
for episode in range(episodes):
    
    print("episode:",episode)
    
    total_reward = 0
    
    obs, info = env.reset(seed=seed+episode)
    
    while True:
            
        action = RL.choose_action(obs)

        obs_, reward, terminated, truncated, info = env.step(action)
        
        RL.cache(obs,obs_,action,reward,terminated)
        
        total_reward += reward
        
        q, loss = RL.learn()

        logger.log_step(reward, loss, q)
        
        obs = obs_
        
        if terminated or truncated:  # 设置最大步数
            break
        
        counter += 1
    logger.log_episode()

    if (episode % 20 == 0) or (episode == episodes - 1):
        logger.record(episode=episode, epsilon=RL.exploration_rate, step=RL.curr_step)
    
env.close()
