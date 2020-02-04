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
    def set_target_lane(self, lane):
        self.target_lane = lane

    def calc(self, pos_info, time, stuck):
        tmp = 23
        width = pos_info[25] + pos_info[26]
        l = pos_info[0:19] + pos_info[23:25]  #sensor, angle, tomid
        l = list(l)
        l = l + [pos_info[tmp + 6], pos_info[tmp + 7], pos_info[tmp + 8], pos_info[50]] #speedX,Y,Z,rpm,23+4+4 = 31

        self.current_length = pos_info[48]

        #goal dist to mid 
        dist_to_mid = [-0.2, -0.1, 0, 0.1, 0.2]
        self.target_lane = 2
        l += [dist_to_mid[self.target_lane-1]]
        # self.cur_pos.append(l[20])
        # self.tar_pos.append(l[-1]*width)
        dist_decay = np.exp(-10 * abs(l[20]/width - l[-1]))

        #reward
        r = (l[21]*np.cos(l[19])) * (dist_decay) - 5
        # if self.time > 600 and l[21] < 10 :
        #     r = -10
        #     stuck = 1

        for i in range(19):
            l[i] = l[i]/200

        # Normalize
        l[19] = l[19]/3.1416   #angle
        l[20] = l[20]/width   #tomiddle
        for i in range(21, 24):
            l[i] = l[i] * 3.6/300
        l[24] = l[24]/10000

        # Constrain
        # print("I'm at", l[20] , "with", np.cos(l[19]*3.1416))
        
        if abs(l[20]) > 0.5 or np.cos(l[19]*3.1416) < -0.1:
            r = -10
            stuck = 1
        
        self.time +=1
        
        l = tuple(l)
        self.last_length = self.current_length
        # if self.time == 2000:
        #     import matplotlib.pyplot as plt
        #     plt.plot(np.arange(len(self.cur_pos)), self.cur_pos)
        #     plt.plot(np.arange(len(self.tar_pos)), self.tar_pos)
        #     plt.savefig("pic.png")
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
        self.where_I_am_last_check = -1
        self.time = 0
        self.stuck = 0
        self.game_over = False
        self.wait_until_loaded()

    def take_action(self, action):
        #waiting to be done: time
        self.conn.send(pack('4f', action[0], action[1], action[2], action[3]))
        
    def observe(self, action_list=[0.0,0.0,0.0,0.0], reset=False):
        recv_data = self.first_receive_data
        obs_dim = 26
        if not reset:
            recv_data = self.conn.recv(1024)

        if not recv_data:
            self.loaded = False
            self.game_over = True
            return np.ones(obs_dim, dtype=np.float32),-1,True,"Gameover"

        try:
            pos_info = unpack('51f', recv_data)
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

















