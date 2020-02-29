import os
from threading import Thread
import random
import time
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from mytorcs.env.mytorcs_sim import MyTorcsController
from mytorcs.env.mytorcs_proc import MyTorcsProcess


class MyTorcsEnv(gym.Env):
    """
    open ai env for torcs
    """

    meta = {
        "render.modes": ["human", "rgb_array"],
    }
    MAX_STEERING_DIFF = 1
    ACTION_NAMES = ["steer", "accelerate", "brake"]
    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0
    ACCELERATE_MIN = 0.0
    ACCELERATE_MAX = 1.0
    BRAKE_MIN=0.0
    BRAKE_MAX=1.0
    VAL_PER_OBS = 26 #obs的维度 

    def __init__(self, frame_skip=2):
        
        print("starting MyTorcs env")
        self.port = ('127.0.0.1', 9999)
        self.viewer = MyTorcsController(time_step=0.05, port=self.port)
         # start simulation subprocess
        self.proc = MyTorcsProcess()
            
        try:
            self.exe_path = "torcs -r /home/v/projects/Self-driving/MyTorcsEnv/torcs/mytorcs/env/practice.xml"
            # self.exe_path = "torcs -r /home/ljf/Desktop/Self-driving/MyTorcsEnv/torcs/mytorcs/env/practice.xml"
        except:
            print("Missing torcs environment var. you must start sim manually")
            self.exe_path = None
        
        #no render
        self.headless = True
        # start simulation com

        # action_space
        self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.ACCELERATE_MIN]),
            high=np.array([self.STEER_LIMIT_RIGHT, self.ACCELERATE_MAX]), dtype=np.float32 )

        # obs data
        self.observation_space = spaces.Box(0, 1, [self.VAL_PER_OBS], dtype=np.float32)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = frame_skip
        #当车子被卡住的时候需要手动reset
        self.need_manual_reset = False
        self.steps = 0

    def set_test(self):
        self.exe_path = "torcs"

    def render(self):
        self.exe_path = "torcs"
        
    def __del__(self):
        self.close()

    def close(self):
        self.proc.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #action的最后一维没有被使用
    def step(self, action, time=[0.0]):
        self.steps += 1
        action_list = [0.0, 0.3, 0.0]
        action_list.extend(time)
        
        #control steer and brake
        action_list[0] = action[0]
        action_list[1] = (action[1]+1)/2
        # action_list[2] = action[2]
    
        for i in range(self.frame_skip):
            #only brake in one of all skip
            if(i != 0):
                action_list[2] = 0.0
            
            if(action_list[2] != 0.0):
                action_list[1] = 0
                
            self.viewer.take_action(action_list)
              
            observation, reward, done, info = self.viewer.observe(action_list)
            # fix broken pipe problem(for manual restart)
            if(info == -1 or done):
                break


        if info == -1:
            done = True
            reward = -10
            
        return observation, reward, done, info

    def reset(self,on_driving_reset=False):
        if self.viewer is None:
            return        
        
        while(not self.proc.quit()):
            print("Error to quit torcs")
            time.sleep(0.5)

        self.proc.start(self.exe_path, headless=self.headless, port=self.port)

        self.viewer.reset()
        
        observation, reward, done, info = self.viewer.observe(reset=True)
        return observation

    def set_target_lane(self, lane):
        self.viewer.set_target_lane(lane)
        


    def is_game_over(self):
        return self.viewer.is_game_over()
