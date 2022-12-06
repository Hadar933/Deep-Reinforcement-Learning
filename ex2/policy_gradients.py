import gym
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import collections
from typing import List

# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make('CartPole-v1')
np.random.seed(1)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

class ValueNetwork:
        def __init__(self, state_size, hidden_layers= [12,12], learning_rate = 0.0004, name='value_network'):
            self.state_size = state_size
            self.learning_rate = learning_rate

            with tf.variable_scope(name):

                self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
                self.value = tf.placeholder(tf.int32, 1, name="value")
                self.R_t = tf.placeholder(tf.float32, name="rewards-to-go")

                tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
                input_size = self.state_size
                layer_input = self.state
                for i in range(len(hidden_layers)-1):
                    self.layers[i].w = tf.get_variable(f"w{i}", [input_size, hidden_layers[i]], initializer=tf2_initializer)
                    self.layers[i].b = tf.get_variable(f"b{i}", [input_size, hidden_layers[i]], initializer=tf2_initializer)
                    self.layers[i].z = tf.add(tf.matmul(layer_input, self.layers[i].w ), self.layers[i].b)
                    layer_input      = tf.nn.relu(self.layers[i].z)
                i += 1
                self.output = tf.layers.dense(layer_input, units=1, activation=None)
                self.loss = tf.reduce_mean((self.R_t - v)**2)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)



def run():
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0004

    render = False

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)
    value_net = ValueNetwork(state_size,hidden_layers=[12,12],learning_rate=learning_rate)

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Compute Rt for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                # TODO: substract the baseline from reward-to-go (do it carefully)
                # At = Rt-Vt
                # see: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt
                feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
            
            # TODO: add training loop for the value network
            # see also explanation in https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients

if __name__ == '__main__':
    run()
