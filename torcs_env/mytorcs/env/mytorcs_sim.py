import socket
import random
from struct import *
import traceback
import collections
import numpy as np
import time
import traceback

class MyTorcsController():
    def __init__(self, time_step=0.05, port=None):
        self.port = port
        self.time_step = time_step
        self.loaded = False
        self.game_over = False
        self.time = 0
        self.stuck = 0
        #for the first obs after reset
        self.first_receive_data = None

        self.cur_pos = []
        self.tar_pos = []
        self.s = socket.socket()
        self.s.bind(port)
        self.s.listen()
        self.last_length=0
        self.current_length=0
        self.target_lane = 0
        self.dist_x = []
        self.dist_y = []
    def set_target_lane(self, lane):
        self.target_lane = lane

    def calc(self, pos_info, time, stuck):
        '''
        l[0:19] -> 19 sensors
        l[19]   ->angle 
        l[20] -> tomid
        '''
        self.dist_x.append(pos_info[-2])
        self.dist_y.append(pos_info[-1])
        tmp = 23
        width = pos_info[25] + pos_info[26]
        l = pos_info[0:19] + pos_info[23:25]  #sensor, angle, tomid
        l = list(l)
        l = l + [pos_info[tmp + 6], pos_info[tmp + 7], pos_info[tmp + 8], pos_info[50]] #speedX,Y,Z,rpm,23+4+4 = 31
        # l += 
        # l += [pos_info[-1]/width]
        # l += [pos_info[-2]/100]
        self.current_length = pos_info[48]

        #goal dist to mid 
        # dist_to_mid = [-0.2, -0.1, 0, 0.1, 0.2]
        # self.target_lane = 
        # l += [self.target_lane / 4]
        # self.cur_pos.append(l[20])
        # self.tar_pos.append(l[-1]*width)
        # if self.time >= 200:
        #     l[-1] +5
        width_decay = np.exp(-10*abs(l[20]/width ))
        # dist_decay = l[-1]*100 - 10 
        # if dist_decay >= 0:
        #     dist_decay = max(0.1, 1 - dist_decay / 100)
        # else:
        #     dist_decay = 0.01
        #reward
        dist_decay = 1
        r = l[21] * np.cos(l[19]) * width_decay * dist_decay
        if 0 <= l[-1]*100 < 5 and abs(l[20] - l[-2]*width < 0.2):
            r = -10
        for i in range(19):
            l[i] = l[i]/200

        # Normalize
        l[19] = l[19]/3.1416   #angle
        l[20] = l[20]/width   #tomiddle
        for i in range(21, 24):
            l[i] = l[i] * 3.6/300
            l[i] = (l[i] + 1) / 2
        l[24] /= 10000
        # Constrain
        # print("I'm at", l[20] , "with", np.cos(l[19]*3.1416))
        
        if abs(l[20]) > 0.5 or np.cos(l[19]*3.1416) < -0.1:
            r = -10
            stuck = 1
        
        self.time +=1
        self.last_length = self.current_length
        # if self.time == 2000:
        #     import matplotlib.pyplot as plt
        #     for i in range(len(self.dist_x)-100):
        #         for j in range(i+1, i+100):
        #             self.dist_x[i] += self.dist_x[j]
        #         self.dist_x[i] /= 100
        #     self.dist_x = self.dist_x[:-100]
        #     plt.ylim([0, 100])
        #     plt.xlim([200, 1500])
        #     plt.title("distance_x")
        #     plt.xlabel("timesteps")
        #     plt.ylabel("distance(km)")
        #     plt.legend()
        #     plt.plot(np.arange(len(self.dist_x)), self.dist_x, label="distance_x")
        #     # plt.plot(np.arange(len(self.dist_y)), self.dist_y, label="distance_y")
        #     plt.savefig("x.png")
 
        l[19] += 0.5
        l[20] = (l[20] + 1) / 2
        l = tuple(np.clip(l, 0, 1))
        return r, l, stuck

    def wait_until_loaded(self):
        while not self.loaded:
            conn, addr = self.s.accept()
            recv_data = conn.recv(1024)
            if recv_data:
                #init setting:
                self.loaded = True
                self.first_receive_data = recv_data
                self.conn = conn
                self.addr = addr
        self.loaded = False

    def reset(self):
        if self.game_over:
            self.game_over = False
            
        self.where_I_am_last_check = -1
        self.time = 0
        self.stuck = 0
        self.wait_until_loaded()

    def take_action(self, action):
        #waiting to be done: time
        self.conn.send(pack('4f', action[0], action[1], action[2], action[3]))
        
    def observe(self, action_list=[0.0,0.0,0.0,0.0], reset=False):
        recv_data = self.first_receive_data
        obs_dim = 27
        if not reset:
            recv_data = self.conn.recv(1024)

        if not recv_data:
            self.loaded = False
            self.game_over = True
            return np.ones(obs_dim, dtype=np.float32),-1,True,"Gameover"

        try:
            pos_info = unpack('53f', recv_data)
        except Exception as e:
            #shutdown
            print("fail to unpack")
            self.loaded = False
            self.game_over = True
            return np.zeros(obs_dim, dtype=np.float32),-1,True,"Gameover"
        
        r, info, self.stuck = self.calc(pos_info, self.time, self.stuck)
        
        if self.stuck:
            pos_info = -1
            self.game_over = True

        return np.array(list(info)), r, self.game_over, pos_info
    
    def is_game_over(self):
        return self.game_over

















