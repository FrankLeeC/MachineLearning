# -*- coding:utf-8 -*-

import copy
import numpy as np
import random

WIDTH = 4
HEIGHT = 3

UP = [-1, 0]
RIGHT = [0, 1]
DOWN = [1, 0]
LEFT = [0, -1]


def random_action():
    a = int(random.random()*4)
    if a == 1:
        return [LEFT, DOWN, UP]
    if a == 2:
        return [UP, LEFT, RIGHT]
    if a == 3:
        return [RIGHT, UP, DOWN]
    return [DOWN, RIGHT, LEFT]


def random_policy():
    m = {} 
    for i in range(HEIGHT):
        for j in range(WIDTH):
            m['%d_%d'%(i, j)] = random_action()
    return m


def refresh_policy(policy, new_action, position):
    i, j = position
    policy['%d_%d'%(i, j)] = new_action


def get_actions(position, policy):
    i, j = position
    return policy['%d_%d'%(i, j)]


def get_other_actions(position, policy):
    return [
        [UP, LEFT, RIGHT],
        [RIGHT, UP, DOWN],
        [DOWN, RIGHT, LEFT],
        [LEFT, UP, DOWN]
    ]


def is_forbidden(position):
    i, j = position
    return i < 0 or i >= HEIGHT or j < 0 or j >= WIDTH 


def get_reward(position):
    return 0.0


def move(current, action):
    new_position = current[0] + action[0], current[1] + action[1]
    if is_forbidden(new_position):
        return current 
    return new_position


def policy_evaluation(old_values, policy):
    while True:
        copy_values = [
        [0, 0, 0, 1],
        [0, 0, 0, -1],
        [0, 0, 0, 0]
        ]
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if (j == 3 and i < 2) or (i == 1 and j == 1):
                    continue
                position = (i, j)
                for n, a in enumerate(get_actions(position, policy)):
                    new_position = move(position, a)
                    i, j = new_position
                    value = old_values[i][j]
                    if n == 0:
                        copy_values[i][j] += 0.8*(get_reward(position) + 0.9 * value)
                    else:
                        copy_values[i][j] += 0.1*(get_reward(position) + 0.9 * value)
        m = np.max(np.asarray(copy_values) - np.asarray(old_values))
        if m < 0.000001:
            break
        old_values = copy.deepcopy(copy_values)
    return old_values


def policy_improvement(values, policy):
    changed = False
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if (j == 3 and i < 2) or (i == 1 and j == 1):
                continue
            position = (i, j)
            new_values = {}
            old_actions = get_actions(position, policy)
            new_actions = get_other_actions(position, policy)
            for k, actions in enumerate(new_actions):
                new_value = 0.0
                for n, a in enumerate(actions):
                    new_position = move(position, a)
                    ni, nj = new_position
                    value = values[ni][nj]
                    if n == 0:
                        new_value += 0.8 * (get_reward(position) + 0.9 * value)
                    else:
                        new_value += 0.1 * (get_reward(position) + 0.9 * value)
                new_values[k] = new_value
            # new_values[3] = values[i][j]
            # new_actions.append(old_actions)
            new_values = dict(zip(new_values.values(), new_values.keys()))
            max_new_value = max(new_values.keys())
            max_actions = new_actions[new_values[max_new_value]]
            if max_actions != old_actions:
                refresh_policy(policy, max_actions, position)
                # print_max_action(position, max_actions, old_actions)
                changed = True
    return changed, policy


def print_max_action(position, max_action, old_actions):
    oa = actions_to_string(old_actions[0])
    ma = actions_to_string(max_action[0])
    print(position, ' ', oa , '->', ma)


def actions_to_string(a):
    if a == LEFT:
        return 'left'
    if a == RIGHT:
        return 'right'
    if a == UP:
        return 'up'
    return 'down' 


def run():
    value = [
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 0, 0]
    ]
    policy = random_policy()
    while True:
        # print('----------------------')
        value = policy_evaluation(value, policy)
        changed, policy = policy_improvement(value, policy)
        if not changed:
            break
    for k, v in policy.items():
        print(k, ':', actions_to_string(v[0]))


def main():
    run()


if __name__ == '__main__':
    main()
