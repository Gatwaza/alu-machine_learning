#!/usr/bin/env python3
"""Training script for Atari Breakout using DQN with keras-rl"""
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from PIL import Image

WINDOW_LENGTH = 4
INPUT_SHAPE = (84, 84)


class AtariProcessor(Processor):
    """Processor to preprocess Atari frames for the DQN agent."""

    def process_observation(self, observation):
        """Process a single observation from the environment.

        Args:
            observation: raw observation from the environment

        Returns:
            processed grayscale resized image as uint8
        """
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        return np.array(img, dtype=np.uint8)

    def process_state_batch(self, batch):
        """Process a batch of states for input to the network.

        Args:
            batch: batch of states to process

        Returns:
            normalized float32 batch
        """
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clip rewards to [-1, 1] range for stable training.

        Args:
            reward: raw reward from environment

        Returns:
            clipped reward
        """
        return np.clip(reward, -1.0, 1.0)


def build_model(nb_actions):
    """Build the CNN model for the DQN agent.

    Args:
        nb_actions: number of possible actions in the environment

    Returns:
        compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Permute((2, 3, 1),
                      input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def main():
    """Main function to train the DQN agent on Atari Breakout."""
    env = gym.make('BreakoutDeterministic-v4')
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    model = build_model(nb_actions)
    model.summary()

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    dqn.compile(Adam(lr=0.00025), metrics=['mae'])

    callbacks = [
        ModelIntervalCheckpoint('dqn_breakout_weights_{step}.h5',
                                interval=250000),
        FileLogger('dqn_breakout_log.json', interval=100)
    ]

    dqn.fit(
        env,
        nb_steps=1750000,
        callbacks=callbacks,
        log_interval=10000,
        visualize=False
    )

    dqn.save_weights('policy.h5', overwrite=True)
    env.close()


if __name__ == '__main__':
    main()
