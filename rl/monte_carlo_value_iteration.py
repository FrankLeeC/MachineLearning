# -*- coding:utf-8 -*-


import copy
import numpy as np
import random

WIDTH = 4
HEIGHT = 3

SQUARE = [
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 0, 0]
]

UP = [-1, 0]
RIGHT = [0, 1]
DOWN = [1, 0]
LEFT = [0, -1]

COPY_SQUARE = None

def reset():
    global COPY_SQUARE
    COPY_SQUARE = [
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 0, 0]
    ]


def valid_positions():
    return [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3]
    ]


def random_start():
    '''
    随机起点
    '''
    positions = valid_positions()
    r = int(random.random() * len(positions))
    return positions[r]


def is_forbidden(position):
    '''
    不合规的位置
    '''
    i, j = position
    return i < 0 or i >= HEIGHT or j < 0 or j >= WIDTH  or (i == j == 1)


def get_reward(position):
    '''
    获取奖励
    '''
    position = list(position)
    if position == [0, 3]:
        return 0
    if position == [0, 2]:
        return -1
    if position == [0, 1]:
        return -2
    if position == [0, 0]:
        return -3
    if position == [1, 3]:
        return -5
    if position == [1, 2]:
        return -2
    if position == [1, 1]:
        return float('-Inf')
    if position == [1, 0]:
        return -4
    if position == [2, 3]:
        return -4
    if position == [2, 2]:
        return -3
    if position == [2, 1]:
        return -4
    if position == [2, 0]:
        return -5


def move(current, action):
    '''
    移动一次
    '''
    new_position = current[0] + action[0], current[1] + action[1]
    if is_forbidden(new_position):
        return current 
    return new_position


def get_actions(position):
    '''
    根据当前位置获取动作
    '''
    i, j = position
    if i == 0:
        if j < 3:
            return RIGHT, True
        return None, False
    if i == 1:
        if j == 0 or j == 2:
            return UP, True
        return None, False
    if i == 2:
        if j == 0 or j == 2:
            return UP, True
        return LEFT, True


def generate_episode(position):
    '''
    生成片段
    '''
    actions = list()
    states = list()
    rewards = list()
    while True:
        action, is_continued = get_actions(position)
        if is_continued:
            states.append(position)
            actions.append(action)
            next_position = move(position, action)
            rewards.append(get_reward(next_position))
            position = next_position
        else:
            break
    return states, actions, rewards


position_returns = {}
state_value = [
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 0, 0]
]


def append_g(position, g):
    global position_returns, state_value
    a, b = position
    key = '%d_%d' % (a, b)
    value = position_returns.get(key, list())
    value.append(g)
    position_returns[key] = value
    print(position, len(value))
    state_value[position[0]][position[1]] = np.mean(value)


def step():
    states, actions, rewards = generate_episode(random_start())
    g = 0
    length = len(states)
    cache = set()
    for i in range(length):
        j = length - i -1
        state = states[j]
        # action = actions[j]
        reward = rewards[j]
        g = 0.9 * g + reward
        a, b = state
        if ('%d_%d' % (a, b)) not in cache:
            cache.add('%d_%d' % (a, b))
            append_g(state, g)
                
            
def run():
    for _ in range(10):
        step()


def main():
    run()
    for _, e in enumerate(state_value):
        print([float('%.2f'%each) for each in e])

if __name__ == '__main__':
    main()
