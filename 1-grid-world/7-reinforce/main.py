import numpy as np
import tkinter as tk
import random
from collections import namedtuple
import tensorflow as tf

# Define the grid world environment
class GridWorldEnv:
    def __init__(self, width=5, height=5, goal=(4, 4), obstacles=((2, 3),)):
        self.width = width
        self.height = height
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

        self.observation_space = np.array([(0, 0), (width, height)]).T
        self.action_space = np.array([0, 1, 2, 3])  # up, down, left, right

    def reset(self):
        self.state = (0, 0)
        self.done = False
        return np.array(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            y = max(y - 1, 0)
        elif action == 1:  # down
            y = min(y + 1, self.height - 1)
        elif action == 2:  # left
            x = max(x - 1, 0)
        elif action == 3:  # right
            x = min(x + 1, self.width - 1)

        self.state = (x, y)
        reward = -1
        if self.state == self.goal:
            reward = 10
            self.done = True
        elif (x, y) in self.obstacles:
            reward = -10
            self.done = True

        return np.array(self.state), reward, self.done

    def render(self, canvas):
        cell_size = 50
        for x in range(self.width):
            for y in range(self.height):
                color = "white"
                if (x, y) == self.goal:
                    color = "green"
                elif (x, y) in self.obstacles:
                    color = "red"
                elif (x, y) == self.state:
                    color = "blue"
                canvas.create_rectangle(x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size, fill=color)
        canvas.update()

# Define the REINFORCE agent
class REINFORCEAgent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99, hidden_layer_size=64):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.hidden_layer_size = hidden_layer_size

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.size  # use 'size' instead of 'n'

        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(self.hidden_layer_size, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(self.action_size, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def act(self, state):
        action_probs = self.model(np.array([state]), training=True)
        action = tf.random.categorical(action_probs, 1)[0, 0]
        return action

    def train(self, episodes=1000, render=False):
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_rewards = []

            while not done:
                action = self.act(state)
                next_state, reward, done = self.env.step(action)

                episode_rewards.append(reward)
                if render:
                    self.env.render(tk.Tk().canvas)

                state = next_state

            discounted_rewards = self.discount_rewards(episode_rewards)
            self.train_on_episode_rewards(discounted_rewards)

            rewards.append(sum(episode_rewards))
            print(f"Episode {episode + 1}: Reward {sum(episode_rewards)}")

        return rewards

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train_on_episode_rewards(self, discounted_rewards):
        with tf.GradientTape() as tape:
            log_probs = []
            states = []
            actions = []

            state = self.env.reset()
            for t in range(len(discounted_rewards)):
                states.append(state)
                action_probs = self.model(np.array([state]), training=True)
                action = tf.random.categorical(action_probs, 1)[0, 0]
                actions.append(action)
                log_prob = tf.math.log(action_probs[0, action])
                log_probs.append(log_prob)
                next_state, _, _ = self.env.step(action)
                state = next_state

            states = np.array(states)
            actions = np.array(actions)
            one_hot_actions = tf.one_hot(actions, self.action_size, dtype=tf.float32)
            log_probs = tf.reshape(tf.stack(log_probs), (-1, 1))  # Reshape log_probs
            loss = -tf.reduce_sum(tf.multiply(log_probs, one_hot_actions))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

if __name__ == "__main__":
    env = GridWorldEnv()
    agent = REINFORCEAgent(env)
    agent.train(episodes=1000, render=False)
