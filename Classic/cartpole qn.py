"""
This solutions aims to solve the cartpole problem with dynamic programming 
approach.
"""
import gym
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make('CartPole-v1')


def build_model(input_shape, output_len, num_hidd_layers=1, neurons_per_layer=12,activation_fn =tf.nn.relu):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neurons_per_layer, input_shape=input_shape,use_bias=False, kernel_regularizer=
    tf.keras.regularizers.l2(l=0.01), kernel_initializer=tf.initializers.random_uniform(0.01, 0.3)))
    for layer in range(num_hidd_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer,activation=activation_fn, kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                     kernel_initializer=tf.initializers.random_uniform(0.01, 0.3)))
    model.add(tf.keras.layers.Dense(output_len, use_bias=False,
                                 kernel_initializer=tf.initializers.random_uniform(0.01,0.03)))
    return model


def get_loss(predicted, true):
    return tf.reduce_sum(tf.square(predicted - true))


def get_next_action(qa, e, reward):
    """
    Returns the preferred action from the set of available actions.
    """
    rand_num = random.random()
    if rand_num < e:
        e = e - e/100
        #print ("Exploring in reward ", reward)
        return env.action_space.sample()
    else:
        return np.argmax(qa)


def print_episode_data(model, episode_data):
    print("Episode result for ", 0)
    print("Previous value:",episode_data["value"][0])
    print("current value:",model.predict(np.array(episode_data["obs"][0])[np.newaxis])[0])


def print_delta(model, episode_data, discount):
    max_len = len(episode_data["obs"])
    ms_delta = 0
    for i in range(max_len):
        state_values = model.predict(np.array(episode_data["obs"][i])[np.newaxis])[0]
        if i < max_len - 1:
            next_state_values = model.predict(np.array(episode_data["next_obs"][i])[np.newaxis])[0]
            delta = episode_data["reward"][i] + discount * np.max(next_state_values)-state_values[episode_data["action"][i]]
            ms_delta += np.square(delta)
            print("delta:",delta," state values:", state_values," Action:",  episode_data["action"][i],
                  " next state values:", next_state_values)
        else:
            delta = episode_data["reward"][i] -state_values[episode_data["action"][i]]
            print("delta:",delta ," state values:", state_values, " Action:",  episode_data["action"][i],
                  " next state values:", episode_data["reward"][i])
            ms_delta += np.square(delta)
    ms_delta = np.sqrt(ms_delta)/(2*max_len)
    print("Mean squared delta:",ms_delta)
    print()


def replay_actions_seq(model, episode_data, discount, replay_count=10):
    max_len = len(episode_data["obs"])
    for j in range(replay_count):
        for i in range(max_len):
            state_values = model.predict(np.array(episode_data["obs"][i])[np.newaxis])[0]
            if i < max_len-1:
                next_state_values = model.predict(np.array(episode_data["next_obs"][i])[np.newaxis])[0]
                #print(i,":delta:", episode_data["reward"][i] + discount * np.max(next_state_values) -
                     # state_values[episode_data["action"][i]])
                state_values[episode_data["action"][i]] = episode_data["reward"][i]+discount*np.max(next_state_values)
                model.fit(np.array(episode_data["obs"][i])[np.newaxis],np.array(state_values)[np.newaxis],verbose=0)
            else:
                #print(i,":delta:", episode_data["reward"][i] -
                     # state_values[episode_data["action"][i]])
                state_values[episode_data["action"][i]] = episode_data["reward"][i]
                model.fit(np.array(episode_data["obs"][i])[np.newaxis], np.array(state_values)[np.newaxis],verbose=0)

    print_delta(model, episode_data, discount)


def replay_actions_batch(model, episode_data, discount, replay_count=10):
    max_len = len(episode_data["obs"])
    train_data = {"obs":[],"target":[]}
    max_range = (max_len/200)*replay_count
    for j in range(replay_count):
        for i in range(max_len):
            state_values = model.predict(np.array(episode_data["obs"][i])[np.newaxis])[0]
            train_data["obs"].append(np.array(episode_data["obs"][i]))
            if i < max_len - 1:
                next_state_values = model.predict(np.array(episode_data["next_obs"][i])[np.newaxis])[0]
                # print(i,":delta:", episode_data["reward"][i] + discount * np.max(next_state_values) -
                # state_values[episode_data["action"][i]])
                state_values[episode_data["action"][i]] = episode_data["reward"][i] + discount * np.max(
                    next_state_values)
                train_data["target"].append(np.array(state_values))
            else:
                # print(i,":delta:", episode_data["reward"][i] -
                # state_values[episode_data["action"][i]])
                state_values[episode_data["action"][i]] = episode_data["reward"][i]
                train_data["target"].append(np.array(state_values))
        model.fit(np.array(train_data["obs"]),np.array(train_data["target"]),verbose=0)
    print_delta(model, episode_data, discount)


def train_model(max_episodes=500):
    """
    Train RF agent   
    """
    tr_model = build_model((env.observation_space.shape[0],),env.action_space.n,num_hidd_layers=3)
    tr_model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                     loss=get_loss
                     )
    es_model = build_model((env.observation_space.shape[0],),env.action_space.n,num_hidd_layers=3)
    es_model.set_weights(tr_model.get_weights())
    es_model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                     loss=get_loss
                     )
    discount = 0.9
    e = 0.1
    episode_data = {"obs": [], "target_qa": [],"value":[],"reward":[],"next_obs":[],"action":[]}
    score = []
    optimizer = tf.train.AdamOptimizer(0.01)
    reached_max = False
    for i_episode in range(max_episodes):
        episode_data["obs"].clear()
        episode_data["target_qa"].clear()
        episode_data["value"].clear()
        episode_data["reward"].clear()
        episode_data["next_obs"].clear()
        episode_data["action"].clear()
        ep_reward = 0
        obs = env.reset()
        done = False
        while not done:
            values = tr_model.predict(np.array(obs)[np.newaxis])[0]
            start_state = obs
            episode_data["value"].append(values)
            episode_data["obs"].append(obs)

            # perform action
            qa_tr = values
            next_action = get_next_action(values, e,ep_reward)
            episode_data["action"].append(next_action)
            obs,reward,done,info = env.step(next_action)
            episode_data["next_obs"].append(obs)
            episode_data["reward"].append(reward)
            ep_reward+=reward

            # evaluate target qa
            q_next = tr_model.predict(np.array(obs)[np.newaxis])[0]
            max_q_next_value = np.max(q_next)
            qa_tr[next_action] = reward + discount * max_q_next_value
            if done == True and ep_reward is not 500:
                qa_tr[next_action] = reward

            episode_data["target_qa"].append(qa_tr)

        print("episode ",i_episode," total score ",ep_reward)
        #print("Batch model before action replay")
        #print_delta(tr_model,episode_data,discount)
        #print("Batch model after action replay")
        if ep_reward != 500:
            replay_actions_batch(tr_model,episode_data,discount,10)
     #   if i_episode % 200 == 0:
      #      es_model.set_weights(tr_model.get_weights())
        #tr_model.fit(np.array(episode_data["obs"]),np.array(episode_data["target_qa"]),epochs=10,verbose=0)
        score.append(ep_reward)
    return score


if __name__=='__main__':
    score = train_model(300)
    plt.plot(score)
    plt.show()

