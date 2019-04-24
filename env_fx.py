import os
import numpy as np
import matplotlib.pyplot as plt

from const import observation_length

class FXEnv:
    def __init__(self, data_path, spread=0.05, min_act_interval=4, load_interval=1):
        
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.name += '_{0}_{1}'.format(load_interval, min_act_interval)
        self.enable_actions = (0,1,2,3) # keep, switch
        self.history = np.load(data_path)[::load_interval]
        self.spread = spread
        
        # phase setting
        self.min_act_interval = min_act_interval
        self.load_interval = load_interval
        y = 1.0/float(self.min_act_interval)
        self.dp =  np.log(y+1) / float(np.max(np.array(self.enable_actions)))
        
        self.reset()

    def update(self, action):
        """
        action: phase progress step
        """
        if self.terminal: pass
        
        self.screen = self.history[self.time_step-observation_length+1:self.time_step+1]
        
        self.phase += np.exp ( self.dp * action ) - 1
        
        if self.phase >= 1.0:
            
            self.switch = True
            self.phase -= 1.0
            
            price = self.screen[-1]
            gain = price - self.price
            if self.position == 'short' : gain *= -1
            self.reward = gain - self.spread
            
            self.log.append((self.price, price, self.position, self.reward, self.time_step))
            
            self.price = price           
            if self.position == 'long':
                self.position = 'short'
            elif self.position == 'short':
                self.position = 'long'
            
        else:
            self.reward = 0
            self.switch = False
            pass
        
        self.time_step += 1
        if self.time_step >= len(self.history):
            self.terminal = True

    def observe(self):
        return self.screen[np.newaxis, :, np.newaxis], self.reward, self.terminal
    
    def draw(self):
        plt.cla()
        im = plt.plot(self.screen)
        
        cnt = 0
        end_time = 0
        if len(self.log) > 0:
        
            for price_start, price_end, pos, reward, time in self.log:
                past_time = self.time_step - time
                if past_time <= observation_length: 
                    cnt += 1

                    c = 'blue' if pos == 'long' else 'red'
                    end_time = observation_length-past_time
                    plt.plot([end_time, end_time], [price_start,price_end], c='k')

                    if cnt == 1:
                        plt.plot([0, end_time], [price_start,price_start], c=c) 
                        last_time = end_time
                    else:
                        plt.plot([last_time, end_time], [price_start,price_start], c=c) 

            c = 'blue' if self.position == 'long' else 'red'
            im = plt.plot([end_time, observation_length-1], [price_end,price_end], c=c) 
        
        return im
        
    def execute_action(self, action):
        self.update(action)

    def reset(self):
        self.time_step = observation_length-1
        self.price = self.history[self.time_step]
        self.phase = 0
        self.position = 'long'
        self.reward = 0
        self.terminal = False
        self.switch = False
        self.log = []
        self.update(0)


if __name__ == '__main__':
    env = FXEnv()
    print(env.name)