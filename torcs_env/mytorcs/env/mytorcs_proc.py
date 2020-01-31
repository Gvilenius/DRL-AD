import subprocess
import os
import time

class MyTorcsProcess(object):

    def __init__(self):
        self.proc = None

    ## ------ Launch Torcs Env ----------- ##

    def start(self, sim_path, headless=False, port=None):
        # Launch Environment
        try:
            self.proc = subprocess.Popen(sim_path.split(" "))
        except Exception as e:
            print("Fail to excute Torcs subprocess: ",end="")
            print(e)

        

    def quit(self):
        if self.proc is not None:
            try:
                os.kill(self.proc.pid, 9)
                os.system("pkill torcs-bin")
                os.waitpid(self.proc.pid, 1)
                self.proc = None
                return True
               
            except Exception as e:
                print(e)
                return False
        else:
            return True
        
    