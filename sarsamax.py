import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from plot_utils import plot_values

def eps_greedy(state, eps, policy):
    if state in policy:
        if np.random.random() > eps:
            action = np.argmax(policy[state])
        else:
            probs = [0.25, 0.25, 0.25, 0.25]
            action = np.random.choice(np.arange(4), p=probs)
    else:
        probs = [0.25, 0.25, 0.25, 0.25]
        action = np.random.choice(np.arange(4), p=probs)
    return action

def Q_learning(env, num_episodes, alpha, gamma=1.0, eps=0.2):
    Q = defaultdict(lambda: np.zeros(env.nA))
    policy = {}
    for i_episode in range(1, num_episodes+1):
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
        state = env.reset()
        action = policy_eps_greedy(state, eps, Q)
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = policy_eps_greedy(next_state, eps, Q)
            Q[state][action] = Q[state][action] + (alpha * (reward + (gamma * Q[next_state][np.argmax(Q[next_state])]) - Q[state][action]))
            action = next_action
            state = next_state
            if done:
                break
    return Q

env = gym.make('CliffWalking-v0')
print(env.action_space)
print(env.observation_space)

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsamax = Q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])
