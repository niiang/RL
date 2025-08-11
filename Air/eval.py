import os
os.chdir('Air')
import gymnasium as gym
import torch
import ale_py
from RL_brain import DQN
from env_preprocess import SkipFrame,GrayScaleObservation,ResizeObservation
gym.register_envs(ale_py)
from gym.wrappers import FrameStack
env = gym.make('ALE/Asterix-v5',render_mode='human')
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)
RL = DQN(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None)

counter = 0
load_file = torch.load('checkpoints/mario_net_1.chkpt')
RL.net.load_state_dict(load_file['model'])
for i in range(10):
    obs, info = env.reset()
    while True:
        action = RL.eval(obs)

        obs_, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
        
        obs = obs_
        
        counter += 1

RL.save_mode()
env.close()