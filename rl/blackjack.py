# -*- coding:utf-8 -*-

import random


cards = ['ace', 2, 3, 4, 5, 6, 7, 8, 9, 10]


def hit():
    return random.choice(cards)


def usable_ace(s):
    if s + 11 <= 21:
        return True
    return False


def prepare():
    '''
    玩家初始化牌直到 >= 12
    返回当前总和，是否拥有可用的ace
    '''
    s = 0
    usable = False
    while s < 12:
        card = hit()
        if card == 'ace':
            if usable_ace(s):
                s += 11
                usable = True
            else:
                s += 1
        else:
            s += card
    return s, usable


def generate_episode():
    dealer = hit()
    if dealer == 'ace':
        dealer = 11
    s, usable = prepare()
    states = list()
    rewards = list()

    states.append([s, dealer, usable])
    done = False
    while s < 20:
        card = hit()
        # print(card)
        if card == 'ace':
            s += 1
        else:
            s += card
        if s > 21:
            rewards.append(-1)
            done = True
        else:
            rewards.append(0)
            states.append([s, dealer, usable])
    if not done:
        bust = False
        while dealer < 17:
            card = hit()
            if card == 'ace':
                if dealer + 11 > 21:
                    dealer += 1
                else:
                    dealer += 11
            else:
                dealer += card
            if dealer > 21:
                bust = True
        # print('dealer:%d'%dealer)
        if bust:
            rewards.append(1)
        else:
            if dealer > s:
                rewards.append(-1)
            elif dealer < s:
                rewards.append(1)
            else:
                rewards.append(0)
    return states, rewards


state_values = {}
state_counts = {}


if __name__ == "__main__":
    count = 5000
    for _ in range(count):
        states, rewards = generate_episode()
        states = list(reversed(states))
        rewards = list(reversed(rewards))

        g = 0.0
        for state, reward in zip(states, rewards):
            g += 0.9 * reward
            key = '%d_%d_%d' % (state[0], state[1], state[2])
            n = state_counts.get(key, 0)
            old = state_values.get(key, 0.0)
            state_values[key] = old + (g - old) / (n + 1)
            state_counts[key] = n + 1
    for k, v in state_values.items():
        print('%s: %f' % (k, float(v)))
    print("state counts: %d" % len(state_values))
            