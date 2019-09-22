"""
This solutions aims to solve the cartpole problem with dynamic programming 
approach.
"""
import gym
import time
env = gym.make('Pendulum-v0')
import random
import math
import pandas as pd
import seaborn as sns

df = pd.DataFrame()
score = pd.Series()


def get_hash(observation):
    """
    Returns a hash representing the current state of the observation
    """
    return_hash=""
    for _ in observation:
        if _<-1:
            return_hash+='-1'
        elif _<0:
            return_hash+='0'
        elif _<1:
            return_hash+='1'
        else:
            return_hash+='2'

    return return_hash


e = 0.03
step_state_dict = dict()
action_state_map = dict()


def get_next_action(_hash):
    """
    Returns the preferred action from the set of available actions.
    """
    rand_num = random.random()
    if rand_num < e:
        if action_state_map[_hash][0]>action_state_map[_hash][1]:
            return 1
        else:
            return 0
    else:
        if action_state_map[_hash][0]>action_state_map[_hash][1]:
            return 0
        else:
            return 1


def visualize():
    """
    Draws charts from data
    """
    _list = list(list())
    global df
    for _episode in range(1,20):
        observation = env.reset()
        done  = False
        while done!=True:
            _list.append(list(observation))
            action  = env.action_space.sample()
            observation,reward,done,info = env.step(action)
            print(reward)
    df = df.append(_list)
    print(df.head())
    df[0].sort_values().plot.line()
    df[1].sort_values().plot.line()
    df[2].sort_values().plot.line()
    sns.heatmap(df.corr(),annot=True)

    input("press enter to continue")





def train_ref(n):
    """
    Train RF agent   
    """
    global score,e
    for i_episode in range(1,n):
        done = False 
        observation = env.reset()
        env.render()
        current_episode_states = []
        t = 0
        while done is False:
            t += 1
            _hash = get_hash(observation)
            if _hash not in action_state_map:
                action_state_map[_hash]=[1,1,0,0]
            action = get_next_action(_hash)
            pair = (_hash,action)
            current_episode_states.append(pair)
            observation,reward,done,_ = env.step(action)
        score  = score.append(pd.Series([t]),ignore_index=True)
        print(t)
        for tup in current_episode_states:
            action_state_map[tup[0]][tup[1]] = (action_state_map[tup[0]][tup[1]]*action_state_map[tup[0]][tup[1]+2]+t)/(action_state_map[tup[0]][tup[1]+2]+1)
            action_state_map[tup[0]][tup[1]+2]+=1
            t = t-1

if __name__=='__main__':
    print(env.action_space)
    visualize()
 #   train_ref(150)
 #   sns.kdeplot(score)
    input("done")

