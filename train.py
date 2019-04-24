#!/usr/bin/env python
# coding: utf-8

import numpy as np
from env_fx import FXEnv
from agent_lstmdqn import LSTMDQNAgent

if __name__ == "__main__":

    n_epochs = 100

    env = FXEnv(
        '/mnt/Omer/Project/10.AdaptivePacing/dst/npy/USDJPY.npy',
        spread=0.,
        min_act_interval=4,
        load_interval=1440*7)

    agent = LSTMDQNAgent(env.enable_actions, env.name)
    agent.init_model()

    for e in range(n_epochs):
        env.reset()
        state_t_1, reward_t, terminal = env.observe()
        state_t = state_t_1
        action_t = agent.select_action(state_t, agent.exploration)
        env.update(action_t)
        agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

        while not terminal:
            
            print('{0}:{1}'.format(e, env.time_step))

            state_t = state_t_1
            action_t = agent.select_action(state_t, agent.exploration)
            env.update(action_t)
            state_t_1, reward_t, terminal = env.observe()
            agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

            agent.experience_replay()

    agent.save_model()