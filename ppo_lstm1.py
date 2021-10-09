import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, lstm_size, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.actor_lstm_c_buffer = np.zeros((size, lstm_size), dtype=np.float32)
        self.actor_lstm_h_buffer = np.zeros((size, lstm_size), dtype=np.float32)
        self.critic_lstm_c_buffer = np.zeros((size, lstm_size), dtype=np.float32)
        self.critic_lstm_h_buffer = np.zeros((size, lstm_size), dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, actor_lstm_c, actor_lstm_h, critic_lstm_c, critic_lstm_h, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.actor_lstm_c_buffer[self.pointer] = actor_lstm_c
        self.actor_lstm_h_buffer[self.pointer] = actor_lstm_h
        self.critic_lstm_c_buffer[self.pointer] = critic_lstm_c
        self.critic_lstm_h_buffer[self.pointer] = critic_lstm_h
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.actor_lstm_c_buffer,
            self.actor_lstm_h_buffer,
            self.critic_lstm_c_buffer,
            self.critic_lstm_h_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, lstm_c, lstm_h, sizes, lstm_size, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    x = tf.expand_dims(x, 1)
    x, c, h = tf.keras.layers.LSTM(lstm_size, return_state=True)(x, [lstm_c, lstm_h])
    return layers.Dense(units=sizes[-1], activation=output_activation)(x), c, h


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
# @tf.function
def sample_action(observation, actor_lstm_c, actor_lstm_h):
    logits, new_actor_lstm_c, new_actor_lstm_h = actor([observation, actor_lstm_c, actor_lstm_h])
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action, new_actor_lstm_c, new_actor_lstm_h


# Train the policy by maxizing the PPO-Clip objective
# @tf.function
def train_policy(
    observation_buffer, actor_lstm_c_buffer, actor_lstm_h_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor([observation_buffer, actor_lstm_c_buffer, actor_lstm_h_buffer])[0], action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor([observation_buffer, actor_lstm_c_buffer, actor_lstm_h_buffer])[0], action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
# @tf.function
def train_value_function(observation_buffer, critic_lstm_c_buffer, critic_lstm_h_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic([observation_buffer, critic_lstm_c_buffer, critic_lstm_h_buffer])[0]) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)
lstm_size = 128

# True if you want to render the environment
render = False


# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v0")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the buffer
buffer = Buffer(observation_dimensions, lstm_size, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
actor_lstm_c_input = keras.Input(shape=(lstm_size,), dtype=tf.float32)
actor_lstm_h_input = keras.Input(shape=(lstm_size,), dtype=tf.float32)
critic_lstm_c_input = keras.Input(shape=(lstm_size,), dtype=tf.float32)
critic_lstm_h_input = keras.Input(shape=(lstm_size,), dtype=tf.float32)
logits, actor_lstm_c_output, actor_lstm_h_output = mlp(observation_input, actor_lstm_c_input, actor_lstm_h_input, list(hidden_sizes) + [num_actions], lstm_size, tf.tanh, None)
actor = keras.Model(inputs=[observation_input, actor_lstm_c_input, actor_lstm_h_input], outputs=[logits, actor_lstm_c_output, actor_lstm_h_output])
value, critic_lstm_c_output, critic_lstm_h_output = mlp(observation_input, critic_lstm_c_input, critic_lstm_h_input, list(hidden_sizes) + [1], lstm_size, tf.tanh, None)
value = tf.squeeze(
    value, axis=1
)
critic = keras.Model(inputs=[observation_input, critic_lstm_c_input, critic_lstm_h_input], outputs=[value, critic_lstm_c_output, critic_lstm_h_output])

actor.summary()
critic.summary()

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length
observation, episode_return, episode_length = env.reset(), 0, 0
actor_lstm_c = np.zeros((1, lstm_size))
actor_lstm_h = np.zeros((1, lstm_size))
critic_lstm_c = np.zeros((1, lstm_size))
critic_lstm_h = np.zeros((1, lstm_size))


# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # Get the logits, action, and take one step in the environment
        observation = observation.reshape(1, -1)
        logits, action, new_actor_lstm_c, new_actor_lstm_h = sample_action(observation, actor_lstm_c, actor_lstm_h)
        observation_new, reward, done, _ = env.step(action[0].numpy())
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t, new_critic_lstm_c, new_critic_lstm_h = critic([observation, critic_lstm_c, critic_lstm_h])
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, actor_lstm_c, actor_lstm_h, critic_lstm_c, critic_lstm_h, action, reward, value_t, logprobability_t)

        # Update the observation
        observation = observation_new
        actor_lstm_c = new_actor_lstm_c
        actor_lstm_h = new_actor_lstm_h
        critic_lstm_c = new_critic_lstm_c
        critic_lstm_h = new_critic_lstm_h

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic([observation.reshape(1, -1), critic_lstm_c, critic_lstm_h])[0]
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0
            actor_lstm_c = np.zeros((1, lstm_size))
            actor_lstm_h = np.zeros((1, lstm_size))
            critic_lstm_c = np.zeros((1, lstm_size))
            critic_lstm_h = np.zeros((1, lstm_size))

    # Get values from the buffer
    (
        observation_buffer,
        actor_lstm_c_buffer,
        actor_lstm_h_buffer,
        critic_lstm_c_buffer,
        critic_lstm_h_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, actor_lstm_c_buffer, actor_lstm_h_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, critic_lstm_c_buffer, critic_lstm_h_buffer, return_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )


