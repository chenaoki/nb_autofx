from collections import deque

import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
import tensorflow as tf

from const import observation_length, dstdir

class LSTMDQNAgent:

    def __init__(self, enable_actions, environment_name):
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        
        self.minibatch_size = 32
        self.replay_memory_size = 1000
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration = 0.1
        self.lstm_hidden = observation_length*2
        
        self.model_dir = os.path.join(dstdir, self.name)
        self.model_dir = os.path.join(self.model_dir, self.environment_name)
        self.cp_path = os.path.join(self.model_dir, 'cp_.{epoch:02d}-{val_loss:.2f}.hdf5')
        self.model_path = os.path.join(self.model_dir,"{}.ckpt".format(self.environment_name))
        
        if not os.path.exists(self.model_dir):os.makedirs(self.model_dir)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        self.init_model()
        
        self.current_loss = 0.

    def init_model(self):
        self.model = Sequential([
            LSTM(self.lstm_hidden,input_shape=(observation_length, 1)),
            Dense(self.n_actions),
            Activation('linear')
        ])
        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))

    def Q_values(self, state):
        return self.model.predict(state)

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.enable_actions)
        else:
            return self.enable_actions[np.argmax(self.Q_values(state))]

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append((state, action, reward, state_1, terminal))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)
        for j in minibatch_indexes:
            
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)
            y_j = self.Q_values(state_j)[0,:]
            
            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                Q = np.max(self.Q_values(state_j_1))
                y_j[action_j_index] = reward_j + self.discount_factor * Q # NOQA

            state_minibatch.append(state_j[0,:])
            y_minibatch.append(y_j)

        state_minibatch = np.array(state_minibatch)
        y_minibatch = np.array(y_minibatch)
        
        # training
        es_cb = EarlyStopping(
            monitor='val_loss', 
            mode='auto',
            patience=3)
        cp_cb = ModelCheckpoint(
            filepath = self.cp_path, 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='auto')
        self.history = self.model.fit(
            state_minibatch, y_minibatch,
            batch_size=self.minibatch_size,
            epochs=10,
            validation_split=0.1,
            callbacks=[es_cb, cp_cb])
        
        self.current_loss = self.history.history['loss'][-1]

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self):
        self.model.save(self.model_path)
