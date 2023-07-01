import os
import numpy as np
import gym # 0.21.0
import cv2 
try:
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = r"D:\zzm_codes\codes2023\mujoco200_win64"
    os.environ["MUJOCO_PY_MJKEY_PATH"] = os.environ["MUJOCO_PY_MUJOCO_PATH"] + "/mjkey.txt"
    
    mjcpy_bin = os.environ["MUJOCO_PY_MUJOCO_PATH"]+"\\bin"
    os.environ["PATH"] += ";" + mjcpy_bin
    os.add_dll_directory(mjcpy_bin) # for mujoco_py in py38
except FileNotFoundError:
    print("Waring: mujoco_py dll has not been loaded")

import mujoco_py
from mujoco_py import GlfwContext
GlfwContext()# offscreen=True

import gym_dmc

print("========== Gym Manipulation Env ==========")
# print(*(gym.envs.registry.env_specs.keys()), sep="\t")
openai_mjc_envnames = [name_i for name_i in gym.envs.registry.env_specs.keys() if "Hand" in name_i] # "Hand" in name_i or 
print(*openai_mjc_envnames, sep="\t")

from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize

if __name__ == "__main__":
    

    # env_name = "Walker-walk-v1"
    env_name = "FetchPush-v1"
    def make_env(): return gym.make(env_name) # space_dtype=np.float32
    test_env = DummyVecEnv([make_env for i in range(4)])
    test_episode = 10; n_timesteps = 50


    video_folder = "log_videos/"
    test_env = VecVideoRecorder(test_env, video_folder, record_video_trigger=lambda x: x == 0, video_length=test_episode * n_timesteps, name_prefix="%s_%s"%(env_name, "randpoli"))

    for epi_i in range(test_episode):
        obs = test_env.reset()
        episode_reward = 0.0
        for _ in range(n_timesteps):
            # in DNN
            if isinstance(obs, dict): 
                obs = np.hstack([obs['achieved_goal'], obs['desired_goal'], obs['observation']]) # for HER
            action = np.random.randn(*test_env.action_space.shape) # random policy
            obs, reward, done, infos = test_env.step(action)
            episode_reward += reward.mean()
            test_env.render("human")

        # end an epoch
        print("[episode %02d] avg reward: %d / %d"%(epi_i, episode_reward, n_timesteps))

        test_env.close()