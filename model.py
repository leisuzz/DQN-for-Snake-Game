import numpy as np
# import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
# from tqdm import tqdm
import os
from game import SnakeGameAI, Direction, Point

# from PIL import Image
# import cv2


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = [-200]

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Agent class
class DQNAgent:
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Main model
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3),
                         input_shape=self.input_size))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(self.output_size, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class DQTrainer:
    def __int__(self, model, discount):
        self.model = model
        self.DISCOUNT = discount
        self.target_model = model

        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    def train_step(self, state, action, reward, next_state, done):

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)
            next_state = tf.expand_dims(next_state, axis=0)
            action = tf.expand_dims(action, axis=0)
            reward = tf.expand_dims(reward, axis=0)
            done = (done, )

        pred = self.model.predict(state)

        target = self.target_model.predict(state)

        X = []
        y = []
        for idx in range(len(done)):

            if not done[idx]:
                max_future_q = np.max(next_state[idx])
                new_q = reward[idx] + self.DISCOUNT * max_future_q
            else:
                new_q = reward[idx]
            target[idx][np.argmax(action[idx]).item()] = new_q

            # And append to our training data
            X.append(state)
            y.append(target)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(X, y, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if done else None)

        # Update target network counter every episode
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(state)


# agent = DQNAgent()
#
# Iterate over episodes
# for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
#
#     # Update tensorboard step every episode
#     agent.tensorboard.step = episode
#
#     # Restarting episode - reset episode reward and step number
#     episode_reward = 0
#     step = 1
#
#     # Reset environment and get initial state
#     current_state = env.reset()
#
#     # Reset flag and start iterating until episode ends
#     done = False
#     while not done:
#
#         # This part stays mostly the same, the change is to query a model for Q values
#         if np.random.random() > epsilon:
#             # Get action from Q table
#             action = np.argmax(agent.get_qs(current_state))
#         else:
#             # Get random action
#             action = np.random.randint(0, env.ACTION_SPACE_SIZE)
#
#         new_state, reward, done = env.step(action)
#
#         # Transform new continous state to new discrete state and count reward
#         episode_reward += reward
#
#         if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
#             env.render()
#
#         # Every step we update replay memory and train main network
#         agent.update_replay_memory((current_state, action, reward, new_state, done))
#         agent.train(done, step)
#
#         current_state = new_state
#         step += 1
#
#     # Append episode reward to a list and log stats (every given number of episodes)
#     ep_rewards.append(episode_reward)
#     if not episode % AGGREGATE_STATS_EVERY or episode == 1:
#         average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
#         min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
#         max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
#         agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
#
#         # Save model, but only when min reward is greater or equal a set value
#         if min_reward >= MIN_REWARD:
#             agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
#
#     # Decay epsilon
#     if epsilon > MIN_EPSILON:
#         epsilon *= EPSILON_DECAY
#         epsilon = max(MIN_EPSILON, epsilon)
