"""
This solutions aims to solve the cartpole problem with dynamic programming 
approach.
"""
import gym
import random
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
env = gym.make('FrozenLake-v0')



def build_q_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_shape=(16,), use_bias=False,
                            kernel_initializer = tf.initializers.random_uniform(minval=0,maxval=0.1,
                                                                              dtype=tf.dtypes.float32))
])
    return model


def get_next_action(action_list, e):
    """
    Returns the preferred action from the set of available actions.
    """
    rand_num = random.random()
    action = np.argmax(action_list)
    if rand_num < e:
        e -= 10 ** -3
        return env.action_space.sample(),e
    else:
        return action,e


def get_model_weights(model):
    i = 0
    model_weights = []
    for layer in model.layers:
        model_weights.append(layer.get_weights())
    return model_weights


def print_model(model):
    model_weights = get_model_weights(model)
    for layer in model_weights:
        print("layer weights:",layer)

def get_loss(true, predicted):
    return tf.reduce_sum(tf.square(true - predicted))

def get_observation(observation):
    obs = np.zeros((1, 16), dtype=np.int32)
    obs[0][observation] = 1
    return obs






def train_model(num_episodes):
    """
    Train RF agent   
    """
    discount = 0.9
    sess = tf.Session()
    finished_list= list()
    e = 0.1
    current_model = build_q_network()
    estimate_model = build_q_network()
    estimate_model.set_weights(current_model.get_weights())
    current_model.compile(
        optimizer=tf.train.GradientDescentOptimizer(0.05),
        loss=get_loss,
        metrics=["mse"]
    )
    reward = 0
    initial_weight = None
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i_episode in range(0, num_episodes):
            print("episode_id ", +i_episode)
            num_episodes += 1
            done = False
            observation = env.reset()
            num_epocs = 1
            episode_train_data = {"obs": [], "target_qa": []}
            while not done:
                # get action to perform for current state
                one_h_obs_old = get_observation(observation)
                train_obs = np.zeros(16, dtype=np.int32)
                train_obs[observation] = 1
                episode_train_data["obs"].append(train_obs)
                curr_state_qa = current_model.predict(one_h_obs_old)[0]
                next_action, e = get_next_action(curr_state_qa, e)

                # get next observation
                observation, reward, done, info = env.step(next_action)
                if done:
                    if reward == 1:
                        print('Episode {} was successful, Agent reached the Goal'.format(i_episode))
                        finished_list.append(1)
                    else:
                        finished_list.append(0)

                one_h_obs = get_observation(observation)
                next_state_qa = estimate_model.predict(one_h_obs)[0]
                if done == False:
                    curr_state_qa[next_action] = reward + discount * np.max(next_state_qa)
                else:
                    curr_state_qa[next_action] = reward

                episode_train_data["target_qa"].append(curr_state_qa)
                current_model.fit(np.array(episode_train_data["obs"]), np.array(episode_train_data["target_qa"]),
                              epochs=1, verbose=0)
            if i_episode%200 ==0 :
                estimate_model.set_weights(current_model.get_weights())
    return finished_list
            



if __name__=='__main__':
    finished_list = train_model(800)
    sns.kdeplot(finished_list)
    print("hey")

