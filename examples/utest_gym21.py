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
# GlfwContext()# offscreen=True

print("========== Gym Manipulation Env ==========")
# print(*(gym.envs.registry.env_specs.keys()), sep="\t")
openai_mjc_envnames = [name_i for name_i in gym.envs.registry.env_specs.keys() if "Hand" in name_i] # "Hand" in name_i or 
print(*openai_mjc_envnames, sep="\t")

if __name__ == "__main__":
    #FetchSlide-v1   FetchPickAndPlace-v1    FetchReach-v1   FetchPush-v1    FetchSlideDense-v1      FetchPickAndPlaceDense-v1    FetchReachDense-v1      FetchPushDense-v1
    
    # HandReach-v0    HandManipulateBlockRotateZ-v0   HandManipulateBlockRotateZTouchSensors-v0        HandManipulateBlockRotateZTouchSensors-v1        HandManipulateBlockRotateParallel-v0     HandManipulateBlockRotateParallelTouchSensors-v0 HandManipulateBlockRotateParallelTouchSensors-v1 HandManipulateBlockRotateXYZ-v0  HandManipulateBlockRotateXYZTouchSensors-v0      HandManipulateBlockRotateXYZTouchSensors-v1      HandManipulateBlockFull-v0       HandManipulateBlock-v0  HandManipulateBlockTouchSensors-v0
    
    env=gym.make("HandManipulateBlockRotateXYZ-v0")
    # print("wrapper re-order observations: ", env.env.env.observation_space.keys())
    print("gym wrapped obs dim: ", env.observation_space.shape)
    print("gym wrapped act dim: ", env.action_space.shape)
    test_episode = 20
    for epi_i in range(test_episode):
        print("episode %02d"%(epi_i))
        step_count = 0; done = False
        obs = env.reset()
        while not done: 
            action = np.random.randn(*env.action_space.shape) # random policy
            obs, reward, done, infos = env.step(action) # gym21: 4 outs
            env.render("human") # rgb_array
            # cv2.imshow("img", img[:, :, ::-1])
            # cv2.waitKey(1)
            step_count +=1
            if done: print("ctrl step accu: ", step_count)