# -*- coding: utf-8 -*-

import numpy as np
import random
import time


# player's sum, dealer showing, usable ace, action
COUNTS = np.zeros([10, 10, 2, 2])
Q = np.zeros([10, 10, 2, 2])

# player's sum, dealer showing, usable ace, action
# usable ace 1
# unusable ace 0
# propability of HIT_ACTION
POLICY = np.zeros([10, 10, 2, 2]) + 0.5  # initial propability       

STICK_ACTION = 0
HIT_ACTION = 1


def get_action(state):
    random.seed()
    if random.random() < POLICY[state[0]-12, state[1]-2, state[2], STICK_ACTION]:
        return STICK_ACTION
    return HIT_ACTION


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


def player_action(player_sum, dealer_showing, usable_ace):
    p = POLICY[player_sum-12][dealer_showing-2][usable_ace][STICK_ACTION]
    random.seed()
    if random.random() < p:
        return STICK_ACTION
    return HIT_ACTION


def generate_episode():
    state = random_state()
    action = get_action(state)
    episodes = list()
    if action == STICK_ACTION:
        dealer_sum = dealer(state[1])
        if state[0] < dealer_sum:
            return [[state, STICK_ACTION, -1]]
        elif state[0] == dealer_sum:
            return [[state, STICK_ACTION, 0]]
        else:
            return [[state, STICK_ACTION, 1]]
    else:
        next_state = state
        while action == HIT_ACTION:
            # episodes.append([next_state, action, 0])
            card = random_card()
            new_sum = state[0] + card
            if new_sum > 21:
                episodes.append([next_state, action, -1])
                return episodes
            else:
                episodes.append([next_state, action, 0])
            next_state = [new_sum, state[1], state[2]]            
            action = player_action(new_sum, state[1], state[2])
        dealer_sum = dealer(state[1])
        if next_state[0] < dealer_sum:
            episodes.append([next_state, STICK_ACTION, -1])
        elif next_state[0] == dealer_sum:
            episodes.append([next_state, STICK_ACTION, 0])
        else:
            episodes.append([next_state, STICK_ACTION, 1])
        return episodes


def run():
    count = 50000
    for _ in range(count):
        g = 0.0
        episodes = generate_episode()
        print(len(episodes))
        episodes = reversed(episodes)
        for episode in episodes:
            state, action, reward = episode
            g = 0.9 * g + reward
            Q[state[0]-12][state[1]-2][state[2]][action] += (g - Q[state[0]-12][state[1]-2][state[2]][action]) / (COUNTS[state[0]-12][state[1]-2][state[2]][action] + 1)
            COUNTS[state[0]-12][state[1]-2][state[2]][action] += 1
            if action == HIT_ACTION and Q[state[0]-12][state[1]-2][state[2]][STICK_ACTION] > Q[state[0]-12][state[1]-2][state[2]][action]:
                POLICY[state[0]-12][state[1]-2][state[2]][STICK_ACTION] = 1.0
                POLICY[state[0]-12][state[1]-2][state[2]][HIT_ACTION] = 0.0
            if action == STICK_ACTION and Q[state[0]-12][state[1]-2][state[2]][STICK_ACTION] < Q[state[0]-12][state[1]-2][state[2]][action]:
                POLICY[state[0]-12][state[1]-2][state[2]][STICK_ACTION] = 0.0
                POLICY[state[0]-12][state[1]-2][state[2]][HIT_ACTION] = 1.0
    for player_sum in range(10):
        for dealer_showing in range(10):
            for usable_ace in range(2):
                for action in range(2):
                    print('player_sum: %d, dealer_showing: %d, usable_ace: %d, action: %d    value: %f' % 
                    (player_sum+12, dealer_showing+2, usable_ace, action, POLICY[player_sum][dealer_showing][usable_ace][action]))


if __name__ == "__main__":
    run()