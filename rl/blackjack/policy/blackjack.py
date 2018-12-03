# -*- coding: utf-8 -*-

import numpy as np
import random
import time

STICK_ACTION = 0
HIT_ACTION = 1


def random_state():
    '''
    return [play's sum, dealer's showing, usable ace]
    play's sum ranges in [12, 21]
    dealer's sum ranges in [2, 11]
    usable ace  0 no  1 yes
    '''
    random.seed()
    r = random.randint(0, 199)
    if r < 100:
        return [int(r / 10) + 12, (r % 10) + 2, 0]
    r -= 100
    return [int(r / 10) + 12, (r % 10) + 1, 1]
    

def random_card():
    random.seed()
    return random.randint(1, 10)


def dealer(showing):
    sum = showing
    while sum < 17:
        card = random_card()
        if card == 1 and sum + card <= 21:
            sum += 11
        else:
            sum += card
    return sum


COUNT = np.zeros([10, 10, 2, 2])
Q = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action 

# 第0表示nousable_ace,第1表示usable_ace
# 第0表示STICK_ACTION, 第1表示HIT_ACTION
POLICY = np.zeros([10, 10, 2, 2]) + 0.5  # initial propability. player's sum, dealer's sum, usable_ace, action 

class State:

    def __init__(self, play_sum, dealer_showing, usable_ace):
        self.player_sum = play_sum
        self.dealer_showing = dealer_showing
        self.usable_ace = usable_ace
        pass
    
    def player_sum(self):
        '''
        [12, 21]
        '''
        return self.play_sum

    def dealer_showing(self):
        '''
        [2, 11]
        '''
        return self.dealer_showing

    def usable_ace(self):
        '''
        0: nousable_ace
        1: usable_ace
        '''
        return self.usable_ace


class Episode:

    def __init__(self):
        self.states = []
        self.action = []
        self.reward = []

    def add_state(self, state):
        self.states.append(state)

    def add_action(self, action):
        self.action.append(action)

    def add_reward(self, reward):
        self.reward.append(reward)
        
    def states(self):
        return self.states

    def action(self):
       return self.action

    def reward(self):
        return self.reward


def random_start_state():
    player_sum = random.randint(12, 21)
    dealer_sum = random.randint(2, 11)
    usable_ace = random.randint(0, 1)
    return State(player_sum, dealer_showing, usable_ace)

def get_action(state):
    '''
    STICK_ACTION 0
    HIT_ACTION   1
    '''
    ps = state.player_sum() - 12
    ds = state.dealer_showing() - 2
    u = state.usable_ace()
    p = POLICY[ps][ds][u]
    if p[0] > p[1]:
        return STICK_ACTION
    return HIT_ACTION

def generate_eposide():
    episode = Episode()
    state = random_start_state()
    action = get_action(state)
    # usable_ace = state.usable_ace()  # 不会改变
    next_state = State(state.player_sum(), state.dealer_showing(), state.usable_ace())
    while action == HIT_ACTION:
        episode.add_state(next_state)
        episode.add_action(action)
        card = random_card()
        new_sum = next_state.player_sum() + card
        if new_sum > 21:  # burst
            episode.add_reward(-1)
            return episode
        else:
            episode.add_reward(0)
        next_state = State(new_sum, next_state.dealer_showing(), next_state.usable_ace())
        action = get_action(next_state)
    episode.add_state(next_state)
    episode.add_action(action)
    dealer_sum = dealer(state.dealer_showing())
    if next_state.player_sum() > dealer_sum:
        episode.add_reward(1)
    elif next_state.player_sum() < dealer_sum:
        episode.add_reward(-1)
    else:
        episode.add_reward(0)
    return episode

def run():
    count = 1000
    for i in range(count):
        eposide = generate_eposide()
        # states = set()
        # g = 0.0
        # for state in range(eposide):
        #     key = '%d_%d_%d' % (state[0], state[1], state[2])
        #     if key not in states:  # 当前episode未出现过该状态
        #         g = 0.9 * g + state[3]

    pass
















class Player:

    def __init__(self):
        self.usable_ace = False
        self.current = 0

    def init(self):
        self.usable_ace = False
        self.current = 0

    def random_start():
        '''
        随机初始化状态
        '''
        player_sum = random.randint(12, 21)
        dealer_sum = random.randint(2, 11)
        usable_ace = random.randint(0, 1)
        self.current = player_sum
        self.usable_ace = usable_ace

    def get_action(self, current_sum, dealer_sum):
        '''
        current_sum: [12, 21]
        dealer_sum: [2, 11]
        STICK_ACTION 0
        HIT_ACTION 1
        '''
        global POLICY, Q, COUNT
        current_sum -= 12
        dealer_sum -= 2
        p = POLICY[current_sum][dealer_sum]
        if self.usable_ace:
            if p[1][0] > p[1][1]:
                return STICK_ACTION
            return HIT_ACTION
        else:
            if p[0][0] > p[0][1]:
                return STICK_ACTION
            return HIT_ACTION

    def hit(self):
        '''
        返回是否存活
        '''
        card = random_card()
        if card == 1:
            new_sum = self.current + 11
            if new_sum > 21:
                new_sum -= 10
                self.current = new_sum
                return True
            else:
                self.current = new_sum
                self.usable_ace = True
                return True
        else:
            new_sum = self.current + card
            if new_sum > 21:  # 拿牌后超过了21点
                if self.usable_ace:  # 如果usable_ace,减去10
                    self.current -= 10
                    self.usable_ace = False
                    self.current = new_sum
                    return True
                else:  # 如果nousable_ace,爆了
                    return False
            else:  # 拿牌后没有超过21点
                self.current = new_sum
                return True

    def add_count(self, play_sum, dealer_sum, action):
        global POLICY, Q, COUNT
        if self.usable_ace:
            COUNT[play_sum][dealer_sum][1][action] += 1
        else:
            COUNT[play_sum][dealer_sum][0][action] += 1

    def q(self, state, action, v):
        global POLICY, Q, COUNT
        s, d, u = state[0], state[1], state[2]
        Q += (v - Q[s][d][u]) / COUNT[s][d][u][action]

    def update_policy(self):
        global POLICY, Q, COUNT
        for s in range(10):
            for d in range(10):
                for u in range(2):
                    stick_value = Q[s][d][u][0]
                    hit_value = Q[s][d][u][1]
                    if stick_value > hit_value:
                        POLICY[s][d][u][0] = 1.0
                        POLICY[s][d][u][1] = 0.0
                    else:
                        POLICY[s][d][u][1] = 1.0
                        POLICY[s][d][u][0] = 0.0

