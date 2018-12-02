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


class Player:

    def __init__(self):
        self.usable_ace = False
        self.Q = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action 
        self.COUNT = np.zeros([10, 10, 2, 2])
        self.current = 0

        # 第0表示nousable_ace,第1表示usable_ace
        # 第0表示STICK_ACTION, 第1表示HIT_ACTION
        self.POLICY = np.zeros([10, 10, 2, 2]) + 0.5  # initial propability. player's sum, dealer's sum, usable_ace, action   

    def get_action(self, current_sum, dealer_sum):
        '''
        current_sum: [12, 21]
        dealer_sum: [2, 11]
        STICK_ACTION 0
        HIT_ACTION 1
        '''
        current_sum -= 12
        dealer_sum -= 2
        p = self.POLICY[current_sum][dealer_sum]
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

    def count(self, play_sum, dealer_sum, action):
        if self.usable_ace:
            self.COUNT[play_sum][dealer_sum][1][action] += 1
        else:
            self.COUNT[play_sum][dealer_sum][0][action] += 1

    def q(self, state, action, v):
        s, d, u = state[0], state[1], state[2]
        self.Q += (v - self.Q[s][d][u]) / self.COUNT[s][d][u][action]

    def update_policy(self):
        for s in range(10):
            for d in range(10):
                for u in range(2):
                    stick_value = self.Q[s][d][u][0]
                    hit_value = self.Q[s][d][u][1]
                    if stick_value > hit_value:
                        self.POLICY[s][d][u][0] = 1.0
                        self.POLICY[s][d][u][1] = 0.0
                    else:
                        self.POLICY[s][d][u][1] = 1.0
                        self.POLICY[s][d][u][0] = 0.0


